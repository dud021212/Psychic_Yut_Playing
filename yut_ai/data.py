"""Bounded CPU replay; no GPU-resident replay buffer or worker processes."""
from collections import deque
from dataclasses import dataclass
import random
import torch
from .runtime import DEFAULT_REPLAY_MIB

@dataclass
class Example:
    state: torch.Tensor
    actions: torch.Tensor
    policy: torch.Tensor
    value: float

    def validate(self, config):
        if self.state.shape != (config.state_dim,):
            raise ValueError('State schema mismatch')
        if self.actions.ndim != 2 or self.actions.shape[1] != config.action_dim or len(self.actions) == 0:
            raise ValueError('Expected nonempty legal-action features')
        if self.policy.shape != (len(self.actions),):
            raise ValueError('Policy/action mismatch')
        if not all(bool(torch.isfinite(x).all()) for x in (self.state, self.actions, self.policy)):
            raise ValueError('Nonfinite features')
        if bool((self.policy < 0).any()) or abs(float(self.policy.sum()) - 1) > 1e-5:
            raise ValueError('Policy must be a probability distribution')
        if not -1 <= self.value <= 1:
            raise ValueError('Value must be in [-1, 1]')

class Replay:
    def __init__(self, config, max_bytes=DEFAULT_REPLAY_MIB * 1024**2, seed=0):
        if max_bytes <= 0:
            raise ValueError('Invalid replay budget')
        self.config, self.max_bytes = config, max_bytes
        self.items, self.bytes = deque(), 0
        self.rng = random.Random(seed)

    def add(self, example):
        example.validate(self.config)
        tensors = [x.detach().to('cpu', dtype=torch.float32).clone()
                   for x in (example.state, example.actions, example.policy)]
        size = sum(x.numel() * x.element_size() for x in tensors) + 8
        if size > self.max_bytes:
            raise ValueError('Single example exceeds replay tensor budget')
        while self.bytes + size > self.max_bytes:
            _, old_size = self.items.popleft()
            self.bytes -= old_size
        self.items.append((Example(*tensors, float(example.value)), size))
        self.bytes += size

    def sample(self, count):
        if count <= 0 or not self.items:
            raise ValueError('Need positive batch size and nonempty replay')
        return [self.items[self.rng.randrange(len(self.items))][0] for _ in range(count)]

def collate(examples, device):
    if not examples:
        raise ValueError('Empty batch')
    batch, width = len(examples), max(len(x.actions) for x in examples)
    actions = torch.zeros(batch, width, examples[0].actions.shape[1])
    legal = torch.zeros(batch, width, dtype=torch.bool)
    policy = torch.zeros(batch, width)
    for i, x in enumerate(examples):
        n = len(x.actions)
        actions[i, :n], policy[i, :n], legal[i, :n] = x.actions, x.policy, True
    return tuple(x.to(device) for x in (
        torch.stack([e.state for e in examples]), actions, legal, policy,
        torch.tensor([e.value for e in examples], dtype=torch.float32)))
