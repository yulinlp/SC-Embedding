"""Full candidate-pool losses with bounded activation memory and RNG replay."""
from contextlib import contextmanager

import torch


def token_batches(sequences, budget):
    start = 0
    while start < len(sequences):
        end, longest = start, 0
        while end < len(sequences):
            proposed = max(longest, len(sequences[end]))
            if end > start and proposed * (end - start + 1) > budget:
                break
            longest, end = proposed, end + 1
        yield start, end
        start = end


def rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type != "cpu":
        state[device.type] = getattr(torch, device.type).get_rng_state(device)
    return state


def set_rng_state(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type != "cpu":
        getattr(torch, device.type).set_rng_state(state[device.type], device)


@contextmanager
def replay_rng(state, device):
    current = rng_state(device)
    set_rng_state(state, device)
    try:
        yield
    finally:
        set_rng_state(current, device)


def cached_backward(encoder, sequences, objective, *, budget=8192, scale=1.):
    batches = list(token_batches(sequences, budget))
    cached, states = [], []
    with torch.no_grad():
        for start, end in batches:
            states.append(rng_state(encoder.device))
            cached.append(encoder(sequences[start:end]))
    leaves = torch.cat(cached).detach().requires_grad_(True)
    loss, metrics = objective(leaves)
    gradient, = torch.autograd.grad(loss * scale, leaves)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError("Nonfinite embedding gradients")
    for (start, end), state, reference in zip(batches, states, cached):
        with replay_rng(state, encoder.device):
            values = encoder(sequences[start:end])
            torch.testing.assert_close(values.detach(), reference, atol=2e-5, rtol=2e-5)
            (values * gradient[start:end]).sum().backward()
    return metrics
