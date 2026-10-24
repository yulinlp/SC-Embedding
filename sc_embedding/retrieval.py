"""Causal BM25/cosine reciprocal rank fusion, without query or label inputs."""
from collections import Counter, defaultdict
import math
import re

import numpy as np


def select_history(texts, embeddings, top_k=4, candidate_k=32, rrf_constant=60):
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().float().cpu().numpy()
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or len(texts) != len(vectors) or not texts:
        raise ValueError("Expected aligned, ordered passages and embeddings")
    norms = np.sqrt(np.sum(vectors * vectors, axis=1, dtype=np.float64))
    if not np.isfinite(vectors).all() or not (norms > 0).all():
        raise ValueError("Embeddings must be finite and nonzero")
    if not 0 <= top_k <= candidate_k or rrf_constant <= 0:
        raise ValueError("Invalid retrieval configuration")
    vectors = vectors / norms[:, None]
    documents = [Counter(re.findall(r"\w+(?:[.-]\w+)*", t.casefold())) for t in texts]
    lengths = np.asarray([sum(d.values()) for d in documents], dtype=np.float64)
    postings = defaultdict(lambda: ([], []))
    total_length = 0.
    selected = []
    for current, document in enumerate(documents):
        if not current or not top_k:
            selected.append([])
        else:
            dense = np.sum(vectors[:current] * vectors[current], axis=1, dtype=np.float64)
            sparse = np.zeros(current, dtype=np.float64)
            average = total_length / current or 1.
            for term in sorted(document):
                ids, frequencies = postings.get(term, ([], []))
                if not ids:
                    continue
                ids = np.asarray(ids, dtype=np.int64)
                frequencies = np.asarray(frequencies, dtype=np.float64)
                idf = math.log(1 + (current - len(ids) + .5) / (len(ids) + .5))
                sparse[ids] += idf * frequencies * 2.2 / (
                    frequencies + 1.2 * (.25 + .75 * lengths[ids] / average))
            dense, sparse = np.round(dense, 12), np.round(sparse, 12)
            dense_order = np.lexsort((np.arange(current), -dense))[:candidate_k]
            sparse_ids = np.flatnonzero(sparse > 0)
            sparse_order = sparse_ids[np.lexsort((sparse_ids, -sparse[sparse_ids]))[:candidate_k]]
            fused = {}
            for order in (dense_order, sparse_order):
                for rank, index in enumerate(order, 1):
                    index = int(index)
                    fused[index] = fused.get(index, 0.) + 1 / (rrf_constant + rank)
            selected.append(sorted(sorted(fused, key=lambda i: (-fused[i], i))[:top_k]))
        for term, frequency in document.items():
            postings[term][0].append(current)
            postings[term][1].append(frequency)
        total_length += lengths[current]
    return selected


def pack_history(tokens, selected, end_id, markers, budget=2048):
    """Whole historical passages; delimiters count toward the history budget.

    Role-marker overhead is separate. Empty-history inputs are exactly Local
    inputs. The current passage is never truncated.
    """
    if len(tokens) != len(selected) or budget < 0:
        raise ValueError("Invalid history pool or budget")
    packed, consumed = [], []
    for i, (current, ids) in enumerate(zip(tokens, selected)):
        if not current or current[-1] != end_id:
            raise ValueError("Passages must end in the end-of-turn token")
        if len(set(ids)) != len(ids) or any(not 0 <= j < i for j in ids):
            raise ValueError("History must contain unique strictly preceding indices")
        prefix, actual = [], []
        for j in sorted(ids):
            if len(tokens[j]) > 1 and len(prefix) + len(tokens[j]) <= budget:
                prefix.extend(tokens[j])
                actual.append(j)
        packed.append(list(markers[0]) + prefix + list(markers[1]) + list(current)
                      if prefix else list(current))
        consumed.append(actual)
    return packed, consumed


def history_views(tokens, selected, end_id, markers, budget=2048):
    full, consumed = pack_history(tokens, selected, end_id, markers, budget)
    bare, _ = pack_history(tokens, [[] for _ in tokens], end_id, markers, budget)
    views = {"full": full, "bare": bare}
    seen = {tuple(map(tuple, value)) for value in views.values()}
    # Drop an ordinal among actually consumed passages, without refilling.
    for rank in range(max(map(len, consumed), default=0)):
        reduced = [[j for k, j in enumerate(ids) if k != rank] for ids in consumed]
        packed, actual = pack_history(tokens, reduced, end_id, markers, budget)
        if actual != reduced:
            raise RuntimeError("Drop-one changed the remaining history")
        key = tuple(map(tuple, packed))
        if key not in seen:
            views[f"drop_{rank}"] = packed
            seen.add(key)
    return views
