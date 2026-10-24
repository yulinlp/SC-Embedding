"""One shared LoRA encoder for independent and contextual representations."""
from contextlib import nullcontext
from functools import wraps
import json
from pathlib import Path
import re

import torch
from torch import nn
from torch.nn import functional as F

from .gradcache import token_batches
from .retrieval import pack_history, select_history

QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
HEADERS = ("<history>\n", "</history>\n<passage>\n")
PROJECTIONS = {
    "self_attn": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "linear_attn": {"in_proj_qkv", "out_proj", "in_proj_z", "in_proj_a", "in_proj_b"},
    "mlp": {"gate_proj", "up_proj", "down_proj"},
}


def lora_targets(model):
    pattern = re.compile(r"(?:^|\.)layers\.\d+\.(self_attn|linear_attn|mlp)\.([^.]+)$")
    names = []
    for name, module in model.named_modules():
        match = pattern.search(name)
        if match and isinstance(module, nn.Linear) and match[2] in PROJECTIONS[match[1]]:
            names.append(name)
    if not names:
        raise ValueError("No supported Qwen attention/MLP projections found")
    return sorted(names)


def disable_optional_qwen_kernels():
    # Portable PyTorch path used by the Ascend experiments.
    import transformers.models.qwen3_5.modeling_qwen3_5 as module
    for name in ("causal_conv1d_fn", "causal_conv1d_update", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "FusedRMSNormGated"):
        if hasattr(module, name):
            setattr(module, name, None)


def checkpoint_decoder_layers(model):
    from torch.utils.checkpoint import checkpoint
    def wrap(layer):
        original = layer.forward
        @wraps(original)
        def forward(*args, **kwargs):
            if not torch.is_grad_enabled():
                return original(*args, **kwargs)
            return checkpoint(original, *args, use_reentrant=False, **kwargs)
        layer.forward = forward
    for name, layer in model.named_modules():
        if re.search(r"(?:^|\.)layers\.\d+$", name):
            wrap(layer)


class SCEmbedding(nn.Module):
    def __init__(self, backbone, tokenizer, *, amp=True, history=None):
        super().__init__()
        self.backbone, self.tokenizer, self.amp = backbone, tokenizer, amp
        self.history = history or dict(top_k=4, candidate_k=32, rrf_constant=60, budget=2048)
        self.end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if self.end_id is None or self.end_id == tokenizer.unk_token_id:
            raise ValueError("The tokenizer must provide <|im_end|>")
        self.markers = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in HEADERS]
        self.query_prefix = tokenizer(QUERY_PREFIX, add_special_tokens=False)["input_ids"]

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    @classmethod
    def from_pretrained(cls, base_model, *, adapter=None, device="cpu", amp=True,
                        rank=64, alpha=128, dropout=.1, gradient_checkpointing=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
        disable_optional_qwen_kernels()
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
        backbone = AutoModelForCausalLM.from_pretrained(
            base_model, local_files_only=True, torch_dtype=torch.bfloat16 if amp else torch.float32,
            attn_implementation="eager")
        # Preserve the native runtime's FP32 parameter/buffer storage; matmuls
        # use BF16 autocast. Do not round normalization/recurrent buffers.
        backbone.float()
        if adapter:
            backbone = PeftModel.from_pretrained(backbone, adapter, is_trainable=False)
        else:
            config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                                target_modules=lora_targets(backbone), task_type="CAUSAL_LM",
                                bias="none")
            backbone = get_peft_model(backbone, config)
        for name, parameter in backbone.named_parameters():
            if ".lora_" in name:
                parameter.data = parameter.data.float()
        if gradient_checkpointing:
            checkpoint_decoder_layers(backbone)
        model = cls(backbone.to(device), tokenizer, amp=amp)
        if adapter and (Path(adapter) / "sc_config.json").is_file():
            model.history = json.loads((Path(adapter) / "sc_config.json").read_text())["history"]
        return model.eval() if adapter else model

    def set_stage(self, stage):
        if stage not in ("local", "contextual"):
            raise ValueError(stage)
        probability = .1 if stage == "local" else 0.
        for config in self.backbone.peft_config.values():
            config.lora_dropout = probability
        for module in self.backbone.modules():
            if hasattr(module, "lora_dropout"):
                for dropout in module.lora_dropout.values():
                    if isinstance(dropout, nn.Dropout):
                        dropout.p = probability
        # eval does not freeze parameters: it makes all contextual views deterministic.
        self.train(stage == "local")
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(".lora_" in name)

    def passage_tokens(self, texts):
        return [self.tokenizer(text, add_special_tokens=False)["input_ids"] + [self.end_id] for text in texts]

    def query_tokens(self, text):
        return self.query_prefix + self.passage_tokens([text])[0]

    def forward(self, sequences):
        if not sequences or any(not s for s in sequences):
            raise ValueError("Expected nonempty token sequences")
        length = max(map(len, sequences))
        ids = torch.zeros(len(sequences), length, device=self.device, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, -len(sequence):] = torch.tensor(sequence, device=self.device)
            mask[i, -len(sequence):] = 1
        autocast = torch.autocast(self.device.type, dtype=torch.bfloat16) if self.amp else nullcontext()
        with autocast:
            embeds = self.backbone.get_input_embeddings()(ids).detach()
            embeds = embeds.masked_fill(~mask.bool().unsqueeze(-1), 0.)
            # Match the native Qwen3.5 minimum-length convention.
            if length < 4:
                embeds = F.pad(embeds, (0, 0, 0, 4-length))
                mask = F.pad(mask, (0, 4-length))
            positions = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
            output = self.backbone(inputs_embeds=embeds, attention_mask=mask,
                                   position_ids=positions, output_hidden_states=True,
                                   use_cache=False, logits_to_keep=1)
            return F.normalize(output.hidden_states[-1][:, -1].float(), dim=-1)

    @torch.no_grad()
    def encode_tokens(self, sequences, budget=8192):
        return torch.cat([self(sequences[a:b]) for a, b in token_batches(sequences, budget)])

    @torch.no_grad()
    def encode_queries(self, queries, budget=8192):
        self.eval()
        return self.encode_tokens([self.query_tokens(q) for q in queries], budget)

    @torch.no_grad()
    def encode_passages(self, passages, *, contextual=True, budget=8192):
        """Pass one ordered history stream; do not mix unrelated documents."""
        self.eval()
        tokens = self.passage_tokens(passages)
        local = self.encode_tokens(tokens, budget)
        if not contextual:
            return local
        cfg = self.history
        ids = select_history(passages, local, cfg["top_k"], cfg["candidate_k"], cfg["rrf_constant"])
        sequences, _ = pack_history(tokens, ids, self.end_id, self.markers, cfg["budget"])
        return self.encode_tokens(sequences, budget)

    def save(self, directory, training_config=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.backbone.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)
        (directory / "sc_config.json").write_text(json.dumps(dict(
            history=self.history, query_prefix=QUERY_PREFIX, headers=list(HEADERS),
            pooling="final_hidden_state", normalize=True,
            training=training_config), indent=2) + "\n")
