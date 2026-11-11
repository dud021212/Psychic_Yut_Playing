"""Network input boundary: public features + own powers + inferred belief."""
from dataclasses import dataclass
from math import isfinite
from .belief import PowerBelief

@dataclass(frozen=True)
class PlayerObservation:
    public_features: tuple[float, ...]
    own_powers: tuple[int, int]
    opponent_belief: tuple[float, ...]

    @classmethod
    def from_belief(cls, public_features, belief: PowerBelief):
        return cls(tuple(public_features), belief.own, belief.features())

    def encode(self):
        if len(self.own_powers) != 2 or any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < 8 for i in self.own_powers):
            raise ValueError('Invalid own hand')
        if len(self.opponent_belief) != 64:
            raise ValueError('Invalid belief shape')
        if any(not isfinite(p) or p < 0 for p in self.opponent_belief):
            raise ValueError('Invalid belief probabilities')
        if abs(sum(self.opponent_belief) - 1) > 1e-6:
            raise ValueError('Belief must sum to one')
        for a in range(8):
            for b in range(8):
                if (a == self.own_powers[0] or b == self.own_powers[1]) and self.opponent_belief[a*8+b] != 0:
                    raise ValueError('Impossible duplicate power in belief')
        if any(not isfinite(v) for v in self.public_features):
            raise ValueError('Invalid public features')
        own = tuple(float(i == power) for power in self.own_powers for i in range(8))
        return self.public_features + own + self.opponent_belief
