# Release validation

Validation performed during local release preparation:

- **8 CPU tests passed**, including Qwen3 and Qwen3.5 hybrid-attention models.
- **Two-process CPU launch passed:** the actual `scripts/train.sh` entry point
  completed two Local updates followed by two contextual updates, writing one
  final adapter and no intermediate checkpoint.
- **Inference CLI passed:** reloaded the final tiny adapter and encoded an
  ordered passage stream plus independent queries.
- **Native loss parity:** 40 cases matched the preserved experiment's
  selective objective values and document gradients exactly.
- **Native retrieval parity:** 600 causal passage selections matched the
  experiment's stable semantic/BM25 rank fusion exactly.
- **Selected checkpoint export:** all 372 / 372 / 504 LoRA tensors for
  0.8B / 2B / 4B were preserved bitwise and checked against the corresponding
  full backbone's parameter names and shapes using metadata-only construction.

These checks validate the extracted core, small-model training pipeline,
and weight conversion. They do not constitute a new full-size 8-NPU training
run, a CUDA validation, or a re-evaluation of the selected model scores.
