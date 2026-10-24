"""Encode one ordered passage stream and independent queries."""
import argparse
import json
from pathlib import Path

import torch
from sc_embedding.model import SCEmbedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--input", required=True, help='JSON containing "passages" and "queries"')
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    model = SCEmbedding.from_pretrained(args.base_model, adapter=args.adapter,
                                        device=args.device, amp=not args.device.startswith("cpu"))
    data = json.loads(Path(args.input).read_text())
    passages = model.encode_passages(data["passages"]).cpu()
    queries = model.encode_queries(data["queries"]).cpu()
    with Path(args.output).open("x") as stream:
        json.dump(dict(passage_embeddings=passages.tolist(), query_embeddings=queries.tolist(),
                       scores=(queries @ passages.T).tolist()), stream)


if __name__ == "__main__":
    main()
