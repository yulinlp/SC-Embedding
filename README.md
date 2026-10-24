<div align="center">

# SC-Embedding
### Selective Contextualization for Long-Context Retrieval

![WWW 2027 · Under Review](https://img.shields.io/badge/WWW_2027-Under_Review-315EA8?style=flat-square)
[![Models](https://img.shields.io/badge/🤗_Models-Hugging_Face-FFD21E)](#models)
[![GitHub stars](https://img.shields.io/github/stars/yulinlp/SC-Embedding?style=social)](https://github.com/yulinlp/SC-Embedding)
[![License](https://img.shields.io/badge/License-Apache_2.0-315EA8)](LICENSE)

**[English](README.md) | [中文](README_zh.md)**

[Overview](#overview) · [Results](#results) · [Models](#models) · [Inference](#inference) · [Training](#training)

</div>

**SC-Embedding makes contextualization selective: which history to access, and which information to learn from.** It produces contextual passage embeddings for long-document retrieval, conversational search, and agentic memory, with models at **0.8B, 2B, and 4B** scales.

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Contextual embedding paradigms and SC-Embedding retrieval results across eight benchmarks">
</p>

<a id="overview"></a>

## Overview

Passages in long documents and conversations often depend on earlier information. Existing contextual embedding approaches obtain that information through generated descriptions, continuous text windows, or recurrent latent memory. These routes can lose historical details and incorporate context that does not help represent the current passage.

SC-Embedding introduces two levels of selection:

- **Selective retrieval:** retrieve relevant historical passages in their original form using semantic similarity and BM25, preserving their chronological order.
- **Selective contextual learning:** compare retrieval rankings under full, dropped, and empty history views, and learn from reliable views that improve retrieval through self-distillation.

Training combines **Local pretraining** and **selective contextual learning** in one encoder, producing normalized embeddings for standard vector search.

<details>
<summary><b>Training framework</b></summary>

<p align="center">
  <img src="assets/method.png" width="100%" alt="Local pretraining followed by selective history retrieval and cross-view self-distillation">
</p>

</details>

<a id="results"></a>

## Results

Using **116K retrieval-labeled examples** from EvoTrain-180K, the same source dataset used by EvoEmbedding, SC-Embedding improves retrieval at all three model scales. **SC-Embedding-4B gains 7.49 nDCG@10 points over EvoEmbedding-4B**, while also outperforming the larger KaLM-12B.

**Retrieval.** Macro averages over eight benchmarks: LoCoMo, REALTALK, CovidQA, MLDR, DeepPlanning, LooGLE, NovelQA, and ConvoMem. Scores are percentages; Recall@10 follows the LMEB capped-recall convention.

| Model | Size | Recall@10 | nDCG@10 |
| :--- | ---: | ---: | ---: |
| Qwen3-Embedding-8B | 8B | 69.84 | 57.48 |
| KaLM-Gemma3-12B | 12B | 72.29 | 57.94 |
| EvoEmbedding-0.8B | 0.8B | 69.71 | 55.63 |
| EvoEmbedding-2B | 2B | 71.04 | 57.01 |
| EvoEmbedding-4B | 4B | 72.20 | 58.65 |
| **SC-Embedding-0.8B** | **0.8B** | **72.31** | **59.03** |
| **SC-Embedding-2B** | **2B** | **75.33** | **62.23** |
| **SC-Embedding-4B** | **4B** | **78.17** | **66.14** |

**Downstream generation.** SC-Embedding-4B achieves the highest average accuracy across five tasks, leading on four of them and improving the average over EvoEmbedding-4B by **1.78 points**.

| Model | LoCoMo | LongMemEval-s | PersonaMem-32K | PMME-32K | PMME-128K | Average |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| KaLM-Gemma3-12B | 62.99 | 71.60 | 56.88 | 72.17 | 72.81 | 67.29 |
| EvoEmbedding-4B | 64.48 | 74.20 | 55.35 | **72.68** | 73.60 | 68.06 |
| **SC-Embedding-4B** | **67.92** | **75.80** | **57.56** | 72.42 | **75.49** | **69.84** |

Replacing MiniLM with SC-Embedding also improves LoCoMo generation accuracy by **6.75–8.72 points** across Mem0, A-MEM, LightMem, and MemoryOS.

<a id="models"></a>

## Models

| Model | Backbone | LoRA download |
| :--- | :--- | :--- |
| SC-Embedding-0.8B | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | [🤗 165 MiB](https://huggingface.co/linlp22/SC-Embedding-0.8B) |
| SC-Embedding-2B | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | [🤗 257 MiB](https://huggingface.co/linlp22/SC-Embedding-2B) |
| SC-Embedding-4B | [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | [🤗 504 MiB](https://huggingface.co/linlp22/SC-Embedding-4B) |

Download the adapter together with its matching backbone.

<details>
<summary>Checkpoint provenance</summary>

The released adapters were trained with a fixed initial teacher; this repository's training recipe uses online self-distillation. See the model cards and [checkpoint metadata](checkpoints.json).

</details>

## Installation

Install PyTorch for your accelerator, then install this repository:

```bash
git clone https://github.com/yulinlp/SC-Embedding.git
cd SC-Embedding
pip install -e .
```

Development environment: Python 3.11, PyTorch 2.9.0, Transformers 5.8.1, and PEFT 0.14.0; Ascend also requires torch-npu 2.9.0.post2 and CANN 9.0.0. CUDA entry points are included but have not been validated in a full training run.

<a id="inference"></a>

## Inference

Download a checkpoint and its backbone:

```bash
hf download linlp22/SC-Embedding-4B --local-dir checkpoints/SC-Embedding-4B
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir models/Qwen3-4B-Instruct-2507
```

Encode an ordered passage stream and an independent query:

```python
from sc_embedding import SCEmbedding

model = SCEmbedding.from_pretrained(
    "models/Qwen3-4B-Instruct-2507",
    adapter="checkpoints/SC-Embedding-4B",
    device="cuda:0",
)

passages = [
    "Alice moved to Paris.",
    "She now works at the city hospital.",
]
documents = model.encode_passages(passages)
queries = model.encode_queries(["Where does Alice work?"])
scores = queries @ documents.T
print(scores)
```

On Ascend, import `torch` and `torch_npu`, call `torch.npu.set_device(0)`, and use `device="npu:0"`.

Each passage list must belong to **one document or conversation in chronological order**. Only preceding passages are retrieved as history. For independent text embedding, use `model.encode_passages(passages, contextual=False)`.

<a id="training"></a>

## Training

Use the original `data_chunk_*.json` files from [EvoTrain-180K](https://huggingface.co/datasets/MiG-NJU/EvoTrain-180K), or a JSONL file with ordered passages, a query, and zero-based positive indices:

```json
{"id":"example-1","passages":["Alice lives in Paris.","She works as a doctor.","Bob lives in London."],"query":"Where does Alice live?","positives":[0]}
```

Each example needs at least one positive and one negative passage.

**One command runs both stages and saves the final adapter:**

```bash
DEVICE=npu NPROC=8 bash scripts/train.sh \
  --base-model models/Qwen3-4B-Instruct-2507 \
  --data /path/to/EvoTrain-180K \
  --output runs/sc-4b
```

The final adapter is saved to `runs/sc-4b/final/` and can be loaded with the inference code above.

Use the corresponding backbone to train another model size, or set `DEVICE=cuda` for NVIDIA GPUs. `NPROC` must divide the effective batch size of 16. For optional arguments, run `python -m sc_embedding.train --help`.

## Acknowledgments and License

We build on EvoEmbedding / [EvoTrain-180K](https://huggingface.co/datasets/MiG-NJU/EvoTrain-180K), [Qwen](https://github.com/QwenLM), [Transformers](https://github.com/huggingface/transformers), and [PEFT](https://github.com/huggingface/peft). Project code is released under [Apache-2.0](LICENSE); third-party models and data retain their respective licenses. See [NOTICE](NOTICE).
