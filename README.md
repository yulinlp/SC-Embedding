<div align="center">

# SC-Embedding
### Retrieval Augmented Selective Contextualization<br>for Long-Context Retrieval

[![GitHub stars](https://img.shields.io/github/stars/yulinlp/SC-Embedding?style=social)](https://github.com/yulinlp/SC-Embedding)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)](#installation)
[![License](https://img.shields.io/badge/License-Apache_2.0-315EA8)](LICENSE)

**[English](README.md) | [中文](README_zh.md)**

[Overview](#overview) · [Models](#models) · [Quick Start](#quick-start) · [Training](#training)

</div>

Code for **SC-Embedding**, a family of 0.8B, 2B, and 4B contextual embedding
models that select both **which history to access** and **how to learn from it**.

<a id="overview"></a>

## 💡 Overview

A passage's meaning often depends on earlier parts of a document or conversation.
Yet relevant history is not always useful for representing the current passage.
SC-Embedding makes contextualization selective at two levels:

- **Selective retrieval.** Combine semantic similarity and BM25 to retrieve
  relevant preceding passages, preserving their original text and chronological
  order. Explicit history/passage boundaries distinguish context from the
  passage being represented.
- **Selective contextual learning.** Compare rankings under full, dropped,
  and empty history views. Reliable views that improve retrieval provide
  self-distillation targets. Progressive weighting and gradient calibration
  control the strength of this supervision.

<p align="center">
  <img src="assets/method.png" width="100%" alt="SC-Embedding: Local pretraining followed by selective contextual encoding with history retrieval and cross-view self-distillation">
</p>

A single LoRA encoder supports history retrieval, query encoding, and contextual
passage encoding. The output is a normalized dense vector, compatible with
standard similarity search and existing retrieval pipelines.

<a id="models"></a>

## 📦 Models

| Model | Backbone | Selected adapter |
| :--- | :--- | :--- |
| [SC-Embedding-0.8B](https://huggingface.co/linlp22/SC-Embedding-0.8B) | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 165 MiB |
| [SC-Embedding-2B](https://huggingface.co/linlp22/SC-Embedding-2B) | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 257 MiB |
| [SC-Embedding-4B](https://huggingface.co/linlp22/SC-Embedding-4B) | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | 504 MiB |

Download the selected LoRA adapters from the model links above.
[Checkpoint metadata](checkpoints.json) records their base models, Hub revisions,
and SHA-256 checksums. Base weights are supplied separately.

<details>
<summary><b>Checkpoint provenance</b></summary>

The selected existing local adapters retain their original fixed-initial-teacher
training provenance. The training entry point in this repository implements the
final online cross-view self-distillation algorithm. The adapters are exported
unchanged and are not relabeled as outputs of this entry point.

</details>

<a id="installation"></a>

## 🔧 Installation

Install the PyTorch build for your accelerator first. Ascend also requires
matching `torch_npu` and CANN versions.

```bash
git clone https://github.com/yulinlp/SC-Embedding.git
cd SC-Embedding
python -m pip install -e '.[test]'
```

The development environment uses PyTorch 2.9.0, Transformers 5.8.1,
PEFT 0.14.0, torch-npu 2.9.0.post2, and CANN 9.0.0. The paper training
setup is one server with eight Ascend 910B NPUs.

<a id="quick-start"></a>

## 🚀 Quick Start

Download an adapter and its matching backbone, then load them locally:

```bash
hf download linlp22/SC-Embedding-4B --local-dir checkpoints/SC-Embedding-4B
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
```

```python
import torch
import torch_npu
from sc_embedding.model import SCEmbedding

torch.npu.set_device(0)
model = SCEmbedding.from_pretrained(
    "models/Qwen3-4B-Instruct-2507",
    adapter="checkpoints/SC-Embedding-4B",
    device="npu:0",
)

passages = [
    "Alice moved to Paris.",
    "She now works at the city hospital.",
]
documents = model.encode_passages(passages)
queries = model.encode_queries(["Where does Alice work?"])
scores = queries @ documents.T
```

Each `encode_passages` call takes **one ordered document or conversation**.
Queries are encoded independently. For passages without historical context,
use `model.encode_passages(passages, contextual=False)`.
For CUDA, use `device="cuda:0"` and omit the Ascend imports and device selection.
A file-based interface is available in [scripts/encode.py](scripts/encode.py).

<a id="training"></a>

## 🧪 Training

### 1. Prepare the backbone and data

Supply the matching Qwen backbone and the original `data_chunk_*.json` files
from [EvoTrain-180K](https://huggingface.co/datasets/MiG-NJU/EvoTrain-180K).
Training uses the ordered raw turns and retrieval relevance labels. Model and
data downloads are separate; the training launcher reads local files only.

Custom data can use this JSONL format:

```json
{"id":"example-1","passages":["Alice lives in Paris.","Bob lives in London."],"query":"Where does Alice live?","positives":[0]}
```

Positive indices are zero-based. Each row needs at least one positive and one
negative passage. The final query and labels never enter history retrieval.

### 2. Start both stages with one command

```bash
MODEL_SIZE=4B DEVICE=npu NPROC=8 bash scripts/train.sh \
  --base-model /path/to/Qwen3-4B-Instruct-2507 \
  --data /path/to/EvoTrain-180K \
  --output runs/sc-4b
```

Use `MODEL_SIZE=0.8B` or `MODEL_SIZE=2B` with the corresponding backbone.
`DEVICE=cuda` selects NVIDIA GPUs. The effective batch size is 16.

The run performs **Local pretraining → selective contextual learning** with
one model in memory. At the transition, learned weights are retained and AdamW
is reset. **No Local checkpoint is written or reloaded.** Only the final adapter
is saved at `runs/sc-4b/final/`.

| Setting | 0.8B | 2B | 4B |
| :--- | ---: | ---: | ---: |
| Local learning rate | 5e-5 | 5e-5 | 5e-5 |
| Contextual learning rate | 5e-6 | 2e-5 | 1e-5 |
| LoRA rank / alpha | 64 / 128 | 64 / 128 | 64 / 128 |
| History top-k / token budget | 4 / 2048 | 4 / 2048 | 4 / 2048 |

See [configs/](configs/) for run lengths and all hyperparameters.
`--local-steps` and `--contextual-steps` override stopping points without changing
scheduler horizons. [Training details](docs/TRAINING.md) explain data filtering,
optional training subsets, optimization, and checkpoint behavior.

### 3. Validate the installation

```bash
pytest -q
```

Eight CPU tests cover causal retrieval, history views, distillation, gradient
caching, Qwen3/Qwen3.5 attention, and two-stage training. A two-process CPU
training launch and final-adapter reload also passed. These are small-model
checks; the extracted package has not been rerun as a full-size eight-NPU or
CUDA training experiment. See [validation details](VALIDATION.md).

## 📁 File Structure

```text
assets/          Method figure
configs/         0.8B, 2B, and 4B training configurations
sc_embedding/    Encoder, history retrieval, losses, and two-stage training
scripts/         Training and inference entry points
tests/           Core and small-model integration tests
checkpoints.json Selected adapter identities and checksums
```

## Acknowledgments

Our training setup builds on EvoEmbedding and
[EvoTrain-180K](https://huggingface.co/datasets/MiG-NJU/EvoTrain-180K).
We thank the authors of [Qwen](https://github.com/QwenLM),
[Transformers](https://github.com/huggingface/transformers), and
[PEFT](https://github.com/huggingface/peft).

## License

Project code is licensed under [Apache-2.0](LICENSE). Third-party models and
datasets retain their respective licenses; see [NOTICE](NOTICE).
