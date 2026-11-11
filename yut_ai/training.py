from dataclasses import asdict
from pathlib import Path
import torch
from .model import ModelConfig, PolicyValueNet
from .data import collate

class Trainer:
    def __init__(self, model, device='cpu', lr=3e-4, schema='unassigned'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4,
                                           foreach=False)
        self.step = 0
        self.schema = schema

    def train_batch(self, examples):
        for e in examples:
            e.validate(self.model.config)
        states, actions, legal, target, values = collate(examples, self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits, prediction = self.model(states, actions, legal)
        policy_loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
        value_loss = (prediction - values).square().mean()
        loss = policy_loss + value_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        return {'loss': float(loss.detach()), 'policy': float(policy_loss.detach()),
                'value': float(value_loss.detach()), 'step': self.step}

    def checkpoint(self):
        payload = {'format': 2, 'config': asdict(self.model.config), 'schema': self.schema,
                   'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
                   'step': self.step, 'cpu_rng': torch.get_rng_state()}
        if self.device.type == 'mps':
            payload['mps_rng'] = torch.mps.get_rng_state()
        return payload

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        payload = self.checkpoint()
        torch.save(payload, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device='cpu', schema=None):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['format'] != 2 or (schema is not None and schema != saved['schema']):
            raise ValueError('Checkpoint format/schema mismatch')
        trainer = cls(PolicyValueNet(ModelConfig(**saved['config'])), device, schema=saved['schema'])
        trainer.model.load_state_dict(saved['model'])
        trainer.optimizer.load_state_dict(saved['optimizer'])
        trainer.step = saved['step']
        torch.set_rng_state(saved['cpu_rng'])
        if trainer.device.type == 'mps' and 'mps_rng' in saved:
            torch.mps.set_rng_state(saved['mps_rng'])
        return trainer
