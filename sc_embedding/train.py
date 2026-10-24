"""Two-stage training for SC-Embedding."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

from .data import load_rows
from .gradcache import cached_backward
from .losses import retrieval_loss, selective_loss
from .model import SCEmbedding
from .retrieval import history_views, select_history


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def learning_rate(stage, update):
    horizon = stage["scheduler_steps"]
    warmup = stage.get("warmup_steps", math.ceil(horizon * stage.get("warmup_ratio", 0.)))
    # The first update uses LR=0, matching Transformers warm-up schedules.
    if update < warmup:
        factor = update / max(1, warmup)
    else:
        progress = min(1., (update-warmup) / max(1, horizon-warmup))
        factor = (.1 + .9 * .5 * (1 + math.cos(math.pi * progress))
                  if stage["scheduler"] == "cosine" else 1-progress)
    return stage["learning_rate"] * factor


def row_order(rows, seed, effective_batch=16):
    order = list(range(len(rows)))
    rng = random.Random(seed)
    rng.shuffle(order)
    batches = []
    for start in range(0, len(order), effective_batch * 32):
        pool = sorted(order[start:start + effective_batch * 32],
                      key=lambda i: sum(map(len, rows[i]["passages"])))
        batches.extend(pool[i:i+effective_batch] for i in range(0, len(pool), effective_batch))
    rng.shuffle(batches)
    for batch in batches:
        while len(batch) < effective_batch:
            batch.append(order[(len(batch)-len(order)) % len(order)])
    return batches


def synchronize_gradients(parameters, world):
    if world == 1:
        return
    # Reduce gradients in bounded communication buffers.
    bucket, count = [], 0
    def reduce(items):
        flat = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in items])
        dist.all_reduce(flat)
        flat.div_(world)
        offset = 0
        for p in items:
            p.grad = flat[offset:offset+p.numel()].view_as(p).clone()
            offset += p.numel()
    for parameter in parameters:
        bucket.append(parameter)
        count += parameter.numel()
        if count >= 4_000_000:
            reduce(bucket)
            bucket, count = [], 0
    if bucket:
        reduce(bucket)


def train_row(model, row, stage, config, step, scale):
    passages, positives = row["passages"], row["positives"]
    tokens = model.passage_tokens(passages)
    query = model.query_tokens(row["query"])
    budget = config["batch_tokens"]
    if stage == "local":
        def objective(vectors):
            loss = retrieval_loss(vectors[:-1], vectors[-1], positives, config["temperature"])
            return loss, dict(loss=float(loss.detach()))
        return cached_backward(model, tokens + [query], objective, budget=budget, scale=scale)
    with torch.no_grad():
        local = model.encode_tokens(tokens, budget)
        q = model([query])[0]
        history = config["history"]
        selected = select_history(passages, local, history["top_k"], history["candidate_k"], history["rrf_constant"])
        views = history_views(tokens, selected, model.end_id, model.markers, history["budget"])
        scores = {name: F.normalize(model.encode_tokens(sequences, budget).float(), dim=-1)
                  @ F.normalize(q.float(), dim=-1) for name, sequences in views.items()}
    def objective(vectors):
        return selective_loss(vectors, q, positives, scores, step=step,
                              temperature=config["temperature"], **config["distillation"])
    return cached_backward(model, views["full"], objective, budget=budget, scale=scale)


def train_stages(model, rows, config, output, rank=0, world=1, contextual_ids=None):
    """Keep the same weights in memory; reset AdamW at the stage boundary."""
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        (output / "training_config.json").write_text(json.dumps(config, indent=2)+"\n")
    if world > 1:
        dist.barrier()
    effective = config["effective_batch_size"]
    if effective % world:
        raise ValueError("World size must divide effective batch size")
    accumulation = effective // world
    for name in ("local", "contextual"):
        stage = config[name]
        model.set_stage(name)
        seed_all(stage["seed"] + rank)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=stage["learning_rate"],
                                      weight_decay=stage["weight_decay"], foreach=False)
        active_rows = rows if name == "local" or contextual_ids is None else [r for r in rows if r["id"] in contextual_ids]
        if not active_rows:
            raise ValueError("No training rows remain for this stage")
        order = row_order(active_rows, stage["seed"], effective)
        with (output / f"rank{rank}.jsonl").open("a") as log:
            for update in range(stage["steps"]):
                if update and update % len(order) == 0:
                    order = row_order(active_rows, stage["seed"] + update // len(order), effective)
                optimizer.zero_grad(set_to_none=True)
                metrics = []
                for micro in range(accumulation):
                    index = order[update % len(order)][micro * world + rank]
                    metrics.append(train_row(model, active_rows[index], name, config,
                                             update+1, 1/accumulation))
                synchronize_gradients(parameters, world)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                rate = learning_rate(stage, update)
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.step()
                event = dict(stage=name, step=update+1, learning_rate=rate,
                             gradient_norm=float(norm), rows=metrics)
                log.write(json.dumps(event)+"\n")
                log.flush()
                if rank == 0:
                    print(json.dumps({k:v for k,v in event.items() if k != "rows"}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
    if world > 1:
        dist.barrier()
    if rank == 0:
        model.save(output / "final", config)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-model", required=True, help="Local Qwen model directory")
    parser.add_argument("--data", required=True, help="EvoTrain chunk directory or JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--local-steps", type=int)
    parser.add_argument("--contextual-steps", type=int)
    parser.add_argument("--contextual-ids", help="Optional JSON list of training row IDs")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without loading weights")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for name in ("local", "contextual"):
        override = getattr(args, name+"_steps")
        if override is not None:
            config[name]["steps"] = override
        if not 0 < config[name]["steps"] <= config[name]["scheduler_steps"]:
            raise ValueError("Stage steps must be positive and within the scheduler horizon")
    if not Path(args.base_model).is_dir() or not Path(args.data).exists():
        raise FileNotFoundError("Provide existing local model and data paths")
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if args.device == "npu":
        import torch_npu  # noqa: F401
    device = torch.device("cpu" if args.device == "cpu" else f"{args.device}:{local_rank}")
    if device.type != "cpu":
        getattr(torch, device.type).set_device(device)
    if world > 1:
        dist.init_process_group({"npu":"hccl", "cuda":"nccl", "cpu":"gloo"}[args.device],
                                timeout=timedelta(minutes=30))
    try:
        seed_all(config["local"]["seed"])
        rows = load_rows(args.data)
        config["data_rows"] = len(rows)
        config["row_ids_sha256"] = hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest()
        model = SCEmbedding.from_pretrained(args.base_model, device=device, amp=device.type != "cpu",
                    rank=config["lora_rank"], alpha=config["lora_alpha"], gradient_checkpointing=True)
        model.history = config["history"]
        ids = set(map(str, json.loads(Path(args.contextual_ids).read_text()))) if args.contextual_ids else None
        train_stages(model, rows, config, args.output, rank, world, ids)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
