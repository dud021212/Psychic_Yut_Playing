"""Opponent power posterior from player-visible evidence only.

No true opponent hand is accepted. Action likelihoods must be computed by an
observation-only opponent model; a uniform likelihood adds no evidence.
"""
import math
import random

BELIEF_DIM = 64
BELIEF_SCHEMA = 'opponent-joint-8x8-v1'

class PowerBelief:
    def __init__(self, own_throw: int, own_piece: int):
        if any(not isinstance(x, int) or isinstance(x, bool) or not 0 <= x < 8
               for x in (own_throw, own_piece)):
            raise ValueError('Power indices must be in 0..7')
        self.own = (own_throw, own_piece)
        self.probabilities = [0.0 if a == own_throw or b == own_piece else 1/49
                              for a in range(8) for b in range(8)]

    def update(self, likelihoods):
        values = list(likelihoods)
        if len(values) != 64 or any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('Expected 64 nonnegative finite likelihoods')
        weights = [p * likelihood for p, likelihood in zip(self.probabilities, values)]
        total = sum(weights)
        if total <= 0 or not math.isfinite(total):
            raise ValueError('Evidence incompatible with belief; state unchanged')
        self.probabilities = [w / total for w in weights]

    def observe_used_power(self, category: int, index: int):
        if (not isinstance(category, int) or isinstance(category, bool)
                or category not in (0, 1) or not isinstance(index, int)
                or isinstance(index, bool) or not 0 <= index < 8):
            raise ValueError('Invalid visible power')
        self.update([float((a, b)[category] == index)
                     for a in range(8) for b in range(8)])

    def features(self):
        return tuple(self.probabilities)

    def sample(self, rng: random.Random):
        i = rng.choices(range(64), weights=self.probabilities, k=1)[0]
        return divmod(i, 8)
