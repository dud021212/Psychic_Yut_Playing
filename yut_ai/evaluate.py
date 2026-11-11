"""Frozen checkpoint vs heuristic or another checkpoint, independent randomized games; no training."""
import argparse
import json
import math
import random
import secrets
import io
import tempfile
from contextlib import ExitStack
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import torch
from .model import ModelConfig, PolicyValueNet
from .parallel import initialize_worker, frozen_weights
from pathlib import Path
from .game import Game
from .player import SCHEMA, SEARCH_VERSION, search, rollout_action
from .runtime import DEFAULT_INFERENCE_BACKEND, validate_inference_device
from .runtime import configure, device_for, DEFAULT_THREADS, DEFAULT_DEVICE, DEFAULT_WORKERS
from .training import Trainer

_evaluation_models = None

def initialize_evaluation(payloads, inference_backend="torch"):
    global _evaluation_models
    initialize_worker()
    _evaluation_models = []
    for payload in payloads:
        if payload is None:
            _evaluation_models.append(None)
        elif inference_backend == 'coreml-ne':
            from .coreml_backend import CoreMLPredictor
            _evaluation_models.append(CoreMLPredictor(payload))
        else:
            config, weights = payload
            model = PolicyValueNet(ModelConfig(**config))
            model.load_state_dict(torch.load(io.BytesIO(weights), map_location='cpu', weights_only=True))
            model.eval()
            _evaluation_models.append(model)

def evaluation_job(job):
    return evaluate_game(*_evaluation_models, job)

def evaluate_game(model, opponent, job):
    seed, seat, simulations, rollout_steps = job
    rng, srng, opponent_rng = random.Random(seed), random.Random(seed+999999), random.Random(seed+1999999)
    game = Game.new(rng)
    for _ in range(2000):
        active_model = model if game.player == seat else opponent
        active_rng = srng if game.player == seat else opponent_rng
        if active_model is None:
            action = rollout_action(game, opponent_rng)
        else:
            actions, probabilities, _ = search(game, active_model, active_rng,
                simulations=simulations, depth_turns=None, rollout_steps=rollout_steps)
            action = actions[max(range(len(actions)), key=lambda i: probabilities[i])]
        game.apply(action, rng)
        if game.winner is not None:
            return game.winner == seat
    return None


def evaluation_jobs(seed, games, simulations, rollout_steps):
    rng = random.Random(seed)
    return [(rng.getrandbits(128), rng.randrange(2), simulations, rollout_steps)
            for _ in range(games)]


def previous_pair(checkpoint):
    checkpoint = Path(checkpoint)
    history = checkpoint.parent / (checkpoint.stem + '.history')
    sessions = sorted(history.glob('*/session.json'), key=lambda p: p.parent.name)
    if not sessions:
        raise ValueError(f'No previous-version history found: {history}')
    latest = sessions[-1]
    metadata = json.loads(latest.read_text())
    before, after = latest.parent / 'before.pt', latest.parent / 'after.pt'
    if metadata.get('status') not in ('evaluating', 'complete', 'evaluation_interrupted', 'evaluation_failed') or not before.is_file() or not after.is_file():
        raise ValueError('Latest training has not saved an after.pt; finish training or select explicit checkpoints')
    return str(after), str(before)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('--report', type=Path, help='Save final evaluation JSON')
    opponents = parser.add_mutually_exclusive_group()
    opponents.add_argument('--opponent-checkpoint', help='Frozen opponent model; omit for heuristic')
    opponents.add_argument('--previous', action='store_true', help='Compare after.pt vs before.pt of latest training session')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument('--games', type=int, help='Independent games; default 20')
    counts.add_argument('--groups', type=int, help='Legacy count alias: N means 4*N independent games')
    counts.add_argument('--pairs', type=int, help='Legacy count alias: N means 2*N independent games')
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--rollout-steps', type=int, default=128)
    parser.add_argument('--inference-backend', choices=['torch', 'coreml-ne'], default=DEFAULT_INFERENCE_BACKEND, help='Default: torch CPU inference')
    parser.add_argument('--seed', type=int, help='Evaluation seed; omitted generates a fresh random seed')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    parser.add_argument('--device', choices=['cpu', 'mps', 'auto'], default=DEFAULT_DEVICE)
    args = parser.parse_args()
    games = (args.games if args.games is not None else
             4*args.groups if args.groups is not None else
             2*args.pairs if args.pairs is not None else 20)
    if min(games, args.simulations, args.rollout_steps, args.threads, args.workers) < 1:
        parser.error('counts must be positive')
    if args.previous:
        try:
            args.checkpoint, args.opponent_checkpoint = previous_pair(args.checkpoint)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(json.dumps({'selected_current': args.checkpoint,
                          'selected_previous': args.opponent_checkpoint}), flush=True)
    if args.seed is None:
        args.seed = secrets.randbits(53)
    print(json.dumps({'evaluation_seed': args.seed}), flush=True)
    if args.inference_backend == 'coreml-ne':
        from .coreml_backend import require_coreml
        require_coreml()
    configure(args.threads)
    device = device_for(args.device)
    try:
        validate_inference_device(device, args.workers, args.inference_backend)
    except ValueError as exc:
        parser.error(str(exc))
    model = Trainer.load(args.checkpoint, device, schema=SCHEMA).model
    model.eval()
    opponent = None
    if args.opponent_checkpoint:
        opponent = Trainer.load(args.opponent_checkpoint, device, schema=SCHEMA).model
        opponent.eval()
    wins = losses = truncated = 0
    jobs = evaluation_jobs(args.seed, games, args.simulations, args.rollout_steps)
    pool = None
    interrupted = False
    resources = ExitStack()
    try:
        coreml_paths = None
        if args.inference_backend == 'coreml-ne':
            from .coreml_backend import export_model, CoreMLPredictor
            directory = Path(resources.enter_context(tempfile.TemporaryDirectory(prefix='yut-coreml-eval-')))
            coreml_paths = [export_model(m, directory/f'model-{i}.mlpackage') if m is not None else None
                           for i, m in enumerate((model, opponent))]
            if args.workers == 1:
                model, opponent = [CoreMLPredictor(p) if p else None for p in coreml_paths]

        if args.workers > 1:
            payloads = coreml_paths if coreml_paths is not None else [(asdict(m.config), frozen_weights(m)) if m is not None else None
                        for m in (model, opponent)]
            pool = ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_evaluation, initargs=(payloads, args.inference_backend))
            results = pool.map(evaluation_job, jobs)
        else:
            results = (evaluate_game(model, opponent, job) for job in jobs)
        for progress, outcome in enumerate(results, 1):
            if outcome is None: truncated += 1
            elif outcome: wins += 1
            else: losses += 1
            print(json.dumps({'progress': progress, 'total': len(jobs), 'game_seed': jobs[progress-1][0],
                              'seat': jobs[progress-1][1],
                              'wins': wins, 'losses': losses, 'truncated': truncated}), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        raise
    finally:
        if pool:
            if interrupted and hasattr(pool, 'terminate_workers'):
                pool.terminate_workers()
            else:
                pool.shutdown(wait=True, cancel_futures=interrupted)
        resources.close()
    n = wins+losses
    interval = None
    if n:
        p, z = wins/n, 1.96
        center = (p+z*z/(2*n))/(1+z*z/n)
        radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        interval = [center-radius, center+radius]
    result = {'device': str(device) if args.workers == 1 else 'cpu-workers', 'workers': args.workers, 'threads': args.threads, 'checkpoint': args.checkpoint, 'opponent':args.opponent_checkpoint or 'heuristic-v1','wins':wins,'losses':losses,'truncated':truncated,
                      'win_rate_completed':wins/n if n else None,
                      'approx_wilson_95':interval,
                      'inference_backend': args.inference_backend,
                      'compute_units': 'CPU_AND_NE' if args.inference_backend == 'coreml-ne' else None,
                      'ane_execution_verified': False,
                      'protocol': 'independent-games-v1', 'search_version': SEARCH_VERSION, 'games_requested': len(jobs), 'seed': args.seed, 'simulations': args.simulations,
                      'rollout_steps': args.rollout_steps,
                      'note':'Independent randomized games; Wilson interval uses completed games only.'}
    print(json.dumps(result, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2)+'\n')
        temporary.replace(args.report)


if __name__ == '__main__': main()
