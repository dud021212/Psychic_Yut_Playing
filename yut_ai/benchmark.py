"""Compare fixed-weight, fixed-seed game generation; never writes models."""
import argparse
import copy
import json
import time
import torch
from .game import Game
from .model import ModelConfig, PolicyValueNet
from .player import STATE_DIM, ACTION_DIM
from .parallel import GameWorkers
from .runtime import DEFAULT_INFERENCE_BACKEND

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=8)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--rollout-steps', type=int, default=128)
    parser.add_argument('--inference-backend', choices=['torch', 'coreml-ne'], default=DEFAULT_INFERENCE_BACKEND)
    args = parser.parse_args()
    if min(args.games,args.simulations,args.rollout_steps)<1: parser.error('Positive counts required')
    torch.set_num_threads(1); torch.manual_seed(42)
    model = PolicyValueNet(ModelConfig(STATE_DIM, ACTION_DIM))
    arguments = [(42+i,args.simulations,None,2000,args.rollout_steps) for i in range(args.games)]
    original = Game.clone
    reference = None
    for label, workers_count, legacy_copy in [('deepcopy_serial',1,True),('optimized_serial',1,False),('parallel_2',2,False),('parallel_4',4,False)]:
        Game.clone = (lambda self: copy.deepcopy(self)) if legacy_copy else original
        workers = GameWorkers(workers_count, args.inference_backend)
        try:
            start = time.perf_counter()
            results = workers.generate(model, arguments)
            elapsed = time.perf_counter()-start
            events = [result[0][2] for result in results]
            if reference is None: reference = events
            assert events == reference, 'Game outcomes diverged'
            print(json.dumps({'mode':label,'inference_backend':args.inference_backend,'games':args.games,'seconds':elapsed,
                              'games_per_second':args.games/elapsed,'same_events':True}),flush=True)
        finally:
            workers.close(); Game.clone = original

if __name__ == '__main__': main()
