# Training Reference

[Back to README](../README.md)

**Retrieval Augmented Selective Contextualization for Long-Context Retrieval**

SC-Embedding selects preceding passages through semantic/BM25 retrieval and
learns how to use their raw text through selective self-distillation.
One LoRA encoder supplies independent query/passage representations,
history retrieval, contextual embeddings, and training targets.

## Contents

| Path | Purpose |
|---|---|
| `sc_embedding/retrieval.py` | Strictly causal retrieval, boundary formatting, full/drop/empty views |
| `sc_embedding/losses.py` | Retrieval loss, reliable-view selection, gradient-calibrated distillation |
| `sc_embedding/model.py` | Shared Qwen + LoRA encoder and inference API |
| `sc_embedding/train.py` | Continuous two-stage training |
| `configs/` | 0.8B, 2B, and 4B training configurations |
| `scripts/train.sh` | Single-node distributed training entry point |
| `scripts/encode.py` | Load an adapter and encode text |
| `checkpoints.json` | Selected adapter identities and SHA-256 checksums |

## Installation

Use Python 3.10+ and install the PyTorch build for your accelerator first.
For Ascend, install matching `torch_npu` and CANN versions. The development
environment uses PyTorch 2.9.0, torch-npu 2.9.0.post2, and CANN 9.0.0.

```bash
pip install -e '.[test]'
pytest -q
```

The code supports `npu`, `cuda`, and `cpu` devices. CPU is useful for tests;
the paper training setup is one server with eight Ascend 910B NPUs.
The extracted package is checked with small CPU models; full-size distributed
training and CUDA execution have not been rerun for this release preparation.

## Models and data

Supply a local directory for the corresponding base model:

| Size | Base model | Local LR | Contextual LR |
|---|---|---:|---:|
| 0.8B | `Qwen/Qwen3.5-0.8B` | 5e-5 | 5e-6 |
| 2B | `Qwen/Qwen3.5-2B` | 5e-5 | 2e-5 |
| 4B | `Qwen/Qwen3-4B-Instruct-2507` | 5e-5 | 1e-5 |

Training accepts the original `data_chunk_*.json` files from
`MiG-NJU/EvoTrain-180K`. It reads the ordered raw `meta.turns` and
`meta.evidence_turns`; the final turn is the retrieval query. Rows without
valid positives or without negatives are excluded. Every remaining row uses
its complete passage pool; there are no cross-row negatives.

Alternatively, provide JSONL rows in this format:

```json
{"id":"example-1","passages":["Alice lives in Paris.","She works as a doctor.","Bob lives in London."],"query":"Where does Alice live?","positives":[0]}
```

Indices are zero-based. Passage order defines the available history. The query
and relevance labels are used only for training supervision, never for history
retrieval or passage input construction. Dataset licenses remain those of
the original providers. Model/data downloads are separate from training;
the launcher uses local files only.

## Train both stages with one command

```bash
MODEL_SIZE=4B DEVICE=npu NPROC=8 bash scripts/train.sh \
  --base-model /path/to/Qwen3-4B-Instruct-2507 \
  --data /path/to/EvoTrain-180K \
  --output runs/sc-4b
```

Use `MODEL_SIZE=0.8B` or `MODEL_SIZE=2B` with the matching base model for the
other sizes. `DEVICE=cuda` selects NVIDIA GPUs. The world size must divide
the effective batch size (16). Existing output directories are not overwritten.

The same model stays in memory through both stages:

1. **Local pretraining:** optimize independent passage and query embeddings
   with retrieval supervision and LoRA dropout 0.1.
2. **Selective contextual learning:** retain the learned LoRA weights, reset
   AdamW, disable dropout, and optimize complete-history passage embeddings.
   Independent query/history vectors and full/drop/empty targets are recomputed
   with the current weights without gradients. Reliable improvements enable
   KL self-distillation; its weight ramps up and is calibrated at the document
   embedding gradients.

**There is no stage-one checkpoint save or reload.** The run creates logs and
one final PEFT adapter at `runs/sc-4b/final/`. Intermediate recovery checkpoints
are disabled; interrupting training requires starting again.

The JSON configurations expose all stopping and scheduler settings. The
default local/contextual updates are 3000/100, 5300/600, and 5300/600 for
0.8B, 2B, and 4B. Scheduler horizons remain 7276 and 6954 updates, respectively;
changing the stopping point does not silently compress the learning-rate
schedule. To change run length:

```bash
MODEL_SIZE=2B DEVICE=npu NPROC=8 bash scripts/train.sh \
  --base-model /path/to/Qwen3.5-2B --data /path/to/EvoTrain-180K \
  --output runs/sc-2b --local-steps 5300 --contextual-steps 600
```

For an explicitly defined contextual training subset, add
`--contextual-ids /path/to/train_ids.json`, containing a JSON list of source
row IDs. Without it, both stages use all eligible supplied rows. This launcher
provides a self-contained training recipe; it does not replay the private
experiment orchestration or promise identical published scores.

## Use the selected checkpoints

The three companion model directories contain `adapter_model.safetensors`,
`adapter_config.json`, `sc_config.json`, a model card, and provenance/checksums.
They contain the selected existing weights, with tensor values preserved
exactly during conversion to PEFT. Base weights are obtained separately.

```python
import torch
import torch_npu
from sc_embedding.model import SCEmbedding

torch.npu.set_device(0)
model = SCEmbedding.from_pretrained(
    "/path/to/Qwen3-4B-Instruct-2507",
    adapter="/path/to/SC-Embedding-4B",
    device="npu:0",
)
passages = ["Alice moved to Paris.", "She now works at the city hospital."]
documents = model.encode_passages(passages)
queries = model.encode_queries(["Where does Alice work?"])
scores = queries @ documents.T
```

For Ascend, import `torch_npu` and select the NPU before constructing the model,
or use `scripts/encode.py`, which handles device initialization. Each
`encode_passages` call represents **one ordered document/conversation**, not a
batch of unrelated documents. Use `contextual=False` for independent encoding.

The original manifests of these selected local adapters record a fixed initial
teacher. They are exported unchanged. The training entry point implements the
final online cross-view self-distillation algorithm. These are separately
identified artifacts; the adapters are not relabeled as outputs of that entry point.

## Validation and attribution

Tests cover causal retrieval, no-refill view construction, gradient gating and
calibration, dropout-safe gradient caching, and an actual tiny-model two-stage
run that saves and reloads only the final adapter. Release preparation also
checks exported tensor equality and native-loss/selector parity.

The code uses Hugging Face Transformers and PEFT. Training data and the
retrieval-supervised setup build on EvoEmbedding / EvoTrain-180K. Backbone
models are from Qwen. See [LICENSE](../LICENSE) and [NOTICE](../NOTICE) for attribution.
