import argparse
import json
import time
import torch
from .model import ModelConfig, PolicyValueNet
from .training import Trainer
from .runtime import configure, device_for, DEFAULT_THREADS, DEFAULT_BATCH_SIZE, DEFAULT_DEVICE
from .data import Example
from .belief import PowerBelief
from .observation import PlayerObservation


def main():
    parser = argparse.ArgumentParser(description='Synthetic model validation, NOT Yut self-play')
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps'], default=DEFAULT_DEVICE)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    parser.add_argument('--output', default='checkpoints/synthetic-smoke.pt')
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error('steps and batch-size must be positive')
    configure(args.threads)
    torch.manual_seed(42)
    config = ModelConfig(128 + 16 + 64, 32)
    device = device_for(args.device)
    trainer = Trainer(PolicyValueNet(config), device, schema='synthetic-public-own-belief-v2')
    examples = []
    for i in range(args.batch_size):
        n = 2 + i % 15
        policy = torch.zeros(n)
        policy[0] = 1
        observation = PlayerObservation.from_belief(torch.randn(128).tolist(), PowerBelief(i % 8, (i + 1) % 8))
        examples.append(Example(torch.tensor(observation.encode()), torch.randn(n, 32), policy, 0.5))
    start = time.perf_counter()
    first = trainer.train_batch(examples)
    last = first
    for _ in range(args.steps - 1):
        last = trainer.train_batch(examples)
    if device.type == 'mps':
        torch.mps.synchronize()
    seconds = time.perf_counter() - start
    trainer.save(args.output)
    print(json.dumps({'kind': 'SYNTHETIC_ONLY', 'device': str(device),
                      'threads': args.threads, 'batch_size': args.batch_size,
                      'parameters': sum(p.numel() for p in trainer.model.parameters()),
                      'seconds': seconds, 'first': first, 'last': last,
                      'checkpoint': args.output}, indent=2))

if __name__ == '__main__':
    main()
