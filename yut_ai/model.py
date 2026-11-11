from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class ModelConfig:
    state_dim: int
    action_dim: int
    hidden: int = 128
    observation_schema: str = "public-own-belief-v2"
    def __post_init__(self):
        if min(self.state_dim, self.action_dim, self.hidden) <= 0:
            raise ValueError('Feature dimensions must be positive')

class PolicyValueNet(nn.Module):
    """Scores legal actions using public/own/belief features, never true hidden hands."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden
        self.encoder = nn.Sequential(nn.Linear(config.state_dim, h), nn.ReLU(),
                                     nn.Linear(h, h), nn.ReLU())
        # Separate projections avoid copying state features for each action.
        self.state_policy = nn.Linear(h, h)
        self.action_policy = nn.Linear(config.action_dim, h, bias=False)
        self.score = nn.Linear(h, 1)
        self.value = nn.Sequential(nn.Linear(h, h // 2 or 1), nn.ReLU(),
                                   nn.Linear(h // 2 or 1, 1), nn.Tanh())

    def forward(self, states, actions, legal):
        h = self.encoder(states)
        joint = torch.relu(self.state_policy(h).unsqueeze(1) + self.action_policy(actions))
        logits = self.score(joint).squeeze(-1).masked_fill(~legal, -1e9)
        return logits, self.value(h).squeeze(-1)

    @torch.inference_mode()
    def predict(self, states, actions, legal):
        if not bool(legal.any(dim=1).all()):
            raise ValueError('Nonterminal prediction needs at least one legal action')
        logits, value = self(states, actions, legal)
        return logits.softmax(-1), value
