import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sc_embedding.gradcache import cached_backward
from sc_embedding.losses import retrieval_loss, selective_loss
from sc_embedding.retrieval import history_views, pack_history, select_history
from sc_embedding.train import learning_rate, train_stages


def test_history_is_causal_and_future_invariant():
    texts = ["alpha", "beta", "alpha beta", "future"]
    embeddings = np.array([[1, 0], [0, 1], [.6, .8], [1, 1.]])
    full = select_history(texts, embeddings)
    assert full[:3] == select_history(texts[:3], embeddings[:3])
    assert all(all(j < i for j in ids) for i, ids in enumerate(full))
    assert full == select_history(texts, embeddings)


def test_drop_uses_consumed_blocks_and_never_refills():
    tokens = [[1, 2, 9], [3, 4, 5, 6, 9], [7, 9], [8, 9]]
    selected = [[], [0], [0, 1], [0, 1, 2]]
    full, consumed = pack_history(tokens, selected, 9, ([10], [11]), budget=5)
    assert consumed[3] == [0, 2]
    views = history_views(tokens, selected, 9, ([10], [11]), budget=5)
    assert views["drop_0"][3] == [10, 7, 9, 11, 8, 9]
    assert views["bare"] == tokens
    assert full[0] == tokens[0]
    assert all(v[3][-2:] == tokens[3] for v in views.values())


def test_full_self_target_switches_off_and_matches_retrieval_gradient():
    documents = torch.tensor([[.5, 1.], [1., 0.], [-1., .2]], requires_grad=True)
    query = torch.tensor([1., 0.], requires_grad=True)
    scores = F.normalize(documents.detach(), dim=-1) @ query.detach()
    views = {"full": scores.clone().requires_grad_(), "bare": scores.clone().requires_grad_()}
    loss, metrics = selective_loss(documents, query, [0], views, step=200)
    expected = retrieval_loss(documents, query.detach(), [0])
    torch.testing.assert_close(torch.autograd.grad(loss, documents, retain_graph=True)[0],
                               torch.autograd.grad(expected, documents)[0])
    loss.backward()
    assert metrics["alpha"] == 0 and query.grad is None
    assert all(t.grad is None for t in views.values())


def test_useful_view_distills_with_embedding_gradient_cap():
    documents = torch.tensor([[.5, 1.], [1., 0.], [-1., .2]], requires_grad=True)
    query = torch.tensor([1., 0.])
    views = {"full": torch.tensor([.44, 1., -.98]), "bare": torch.tensor([.2, .7, -.5]),
             "drop_0": torch.tensor([.8, .5, -.8], requires_grad=True)}
    loss, metrics = selective_loss(documents, query, [0], views, step=100)
    assert metrics["best_view"] == "drop_0" and metrics["alpha"] > 0
    assert 0 < metrics["kd_to_retrieval_gradient_ratio"] <= .100001
    loss.backward()
    assert views["drop_0"].grad is None


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 4)
        self.dropout = nn.Dropout(.3)
        self.device = torch.device("cpu")

    def forward(self, sequences):
        return F.normalize(self.linear(self.dropout(torch.tensor(sequences, dtype=torch.float32))), dim=-1)


def test_gradient_cache_replays_dropout_and_query_gradients():
    torch.manual_seed(13)
    model = ToyEncoder()
    reference = copy.deepcopy(model)
    sequences = [[1., 0., 2.], [0., 1., 1.], [1., 2., 0.]]
    rng = torch.get_rng_state()
    values = torch.cat([reference([s]) for s in sequences])
    retrieval_loss(values[:-1], values[-1], [0]).backward()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    cached_backward(model, sequences,
        lambda v: (retrieval_loss(v[:-1], v[-1], [0]), {}), budget=3)
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-5, rtol=1e-5)
    assert torch.equal(after, torch.get_rng_state())


def tiny_model():
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from peft import LoraConfig, get_peft_model
    from sc_embedding.model import SCEmbedding, lora_targets
    tokenizer = Tokenizer(WordLevel({"[UNK]":0, "<|im_end|>":1, "a":2, "b":3, "c":4, "d":5}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", additional_special_tokens=["<|im_end|>"])
    backbone = Qwen3ForCausalLM(Qwen3Config(vocab_size=8, hidden_size=16, intermediate_size=24,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        attn_implementation="eager"))
    backbone = get_peft_model(backbone, LoraConfig(r=2, lora_alpha=4, lora_dropout=.1,
                              target_modules=lora_targets(backbone), task_type="CAUSAL_LM"))
    return SCEmbedding(backbone, tokenizer, amp=False)


def test_two_stages_keep_weights_in_memory_and_save_only_final(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = tiny_model()
    cfg = json.loads((Path(__file__).parents[1]/"configs/4B.json").read_text())
    cfg["effective_batch_size"] = 1
    cfg["batch_tokens"] = 128
    for name in ("local", "contextual"):
        cfg[name].update(steps=2, scheduler_steps=3, warmup_steps=0, learning_rate=.01)
    states = []
    original = model.set_stage
    def record(stage):
        states.append((stage, {k:v.detach().clone() for k,v in model.named_parameters() if '.lora_' in k}))
        original(stage)
    model.set_stage = record
    rows = [dict(id="0", passages=["a b c", "b c d", "a d c"], query="a b", positives=[0])]
    output = tmp_path/"run"
    train_stages(model, rows, cfg, output)
    assert [s[0] for s in states] == ["local", "contextual"]
    assert any(not torch.equal(states[0][1][k], states[1][1][k]) for k in states[0][1])
    assert [p.name for p in output.iterdir() if p.is_dir()] == ["final"]
    assert (output/"final/adapter_model.safetensors").is_file()
    assert model.backbone.peft_config["default"].lora_dropout == 0
    assert len(list((output/"rank0.jsonl").open())) == 4
    # Native PEFT can reload the final adapter without a Local checkpoint.
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    restored = tiny_model()
    state = load_file(str(output/"final/adapter_model.safetensors"))
    result = set_peft_model_state_dict(restored.backbone, state)
    assert not result.unexpected_keys
    actual = get_peft_model_state_dict(restored.backbone)
    assert set(actual) == set(state)
    for k in state:
        assert torch.equal(actual[k], state[k])


def test_scheduler_horizon_is_not_shortened_at_selected_stop():
    cfg = json.loads((Path(__file__).parents[1]/"configs/4B.json").read_text())
    assert learning_rate(cfg["contextual"], 0) == 0
    assert learning_rate(cfg["contextual"], 200) == pytest.approx(1e-5)
    assert learning_rate(cfg["contextual"], 599) > 9e-6


def test_qwen35_hybrid_attention_and_contextual_recomputation():
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model
    from sc_embedding.model import SCEmbedding, lora_targets, disable_optional_qwen_kernels, checkpoint_decoder_layers
    disable_optional_qwen_kernels()
    tokenizer = tiny_model().tokenizer
    config = Qwen3_5TextConfig(vocab_size=8, hidden_size=16, intermediate_size=24,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_num_key_heads=2,
        linear_num_value_heads=4, layer_types=["linear_attention", "full_attention"],
        rope_parameters=dict(rope_type="default", mrope_section=[1,1,0], partial_rotary_factor=1.))
    backbone = Qwen3_5ForCausalLM(config)
    targets = lora_targets(backbone)
    assert any("in_proj_qkv" in t for t in targets) and any("in_proj_b" in t for t in targets)
    backbone = get_peft_model(backbone, LoraConfig(r=2, lora_alpha=4, lora_dropout=.1,
                              target_modules=targets, task_type="CAUSAL_LM"))
    checkpoint_decoder_layers(backbone)
    model = SCEmbedding(backbone, tokenizer, amp=False)
    model.set_stage("contextual")
    vectors = model([[2,3,4,1],[3,4,5,1],[2,4,3,1]])
    retrieval_loss(vectors[:-1], vectors[-1], [0]).backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0)
               for n,p in model.named_parameters() if "linear_attn" in n and ".lora_" in n)
