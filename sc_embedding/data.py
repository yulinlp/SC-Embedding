"""Local training files: EvoTrain JSON chunks or canonical retrieval JSONL."""
import json
from pathlib import Path


def normalize_row(row):
    if "meta" in row:
        turns = row["meta"]["turns"]
        passages, query = turns[:-1], turns[-1]
        positives = row["meta"].get("evidence_turns", [])
    else:
        passages, query, positives = row["passages"], row["query"], row["positives"]
    if not passages or not isinstance(query, str) or any(not isinstance(t, str) for t in passages):
        raise ValueError("Expected ordered passage strings and a query string")
    positives = sorted({i for i in positives if type(i) is int and 0 <= i < len(passages)})
    if not positives or len(positives) == len(passages):
        return None
    return dict(id=str(row.get("id", "")), passages=passages, query=query, positives=positives)


def load_rows(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("data_chunk_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    else:
        files = [path]
    if not files:
        raise ValueError("No data_chunk_*.json files found")
    rows = []
    for file in files:
        with file.open() as stream:
            source = (json.loads(line) for line in stream if line.strip()) if file.suffix == ".jsonl" else json.load(stream)
            for raw in source:
                row = normalize_row(raw)
                if row is not None:
                    if not row["id"]:
                        row["id"] = str(len(rows))
                    rows.append(row)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Training rows must be nonempty with unique IDs")
    return rows
