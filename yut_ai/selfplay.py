"""Run real full-rule local games, train, checkpoint, and resume between games."""
import argparse
from datetime import datetime, timezone
import shutil
import subprocess
import sys
import uuid
import json
from pathlib import Path
import random
import time
import torch
from .game import Game, RULESET
from .parallel import GameWorkers
from .player import STATE_DIM, ACTION_DIM, SCHEMA, SEARCH_VERSION, features, action_features, search
from .model import ModelConfig, PolicyValueNet
from .training import Trainer
from .data import Example, Replay
from .runtime import DEFAULT_INFERENCE_BACKEND, validate_inference_device
from .runtime import (configure, device_for, DEPTH_LIMITS, DEFAULT_THREADS,
                      DEFAULT_BATCH_SIZE, DEFAULT_REPLAY_MIB, DEFAULT_DEVICE, DEFAULT_WORKERS)


def play(model, seed, simulations, depth_turns, max_actions, rollout_steps):
    # Real outcomes are independent from search effort.
    game_rng, search_rng, choice_rng = random.Random(seed), random.Random(seed+1000000007), random.Random(seed+2000000011)
    game = Game.new(game_rng)
    records, events = [], []
    for decision in range(max_actions):
        actions, policy, diagnostic = search(game, model, search_rng, simulations=simulations,
                                             depth_turns=depth_turns, rollout_steps=rollout_steps)
        if len(actions) > 1:
            records.append((game.player, torch.tensor(features(game)),
                            torch.tensor([action_features(game, a) for a in actions]), torch.tensor(policy)))
        index = choice_rng.choices(range(len(actions)), weights=policy, k=1)[0]
        event = game.apply(actions[index], game_rng)
        event['search'] = diagnostic
        events.append(event)
        if game.winner is not None:
            examples = [Example(s, a, pi, 1.0 if player == game.winner else -1.0)
                        for player, s, a, pi in records]
            return game, examples, events
    return game, [], events  # No fabricated draw targets.


def save_run(trainer, replay, path, completed, seed, settings):
    payload = trainer.checkpoint()
    payload['search_version'] = SEARCH_VERSION
    payload['run'] = {'completed': completed, 'seed': seed, 'settings': settings,
                      'replay_rng': replay.rng.getstate(),
                      'examples': [(e.state, e.actions, e.policy, e.value) for e, _ in replay.items]}
    temporary = path.with_suffix('.run.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume(saved):
    if 'run' not in saved:
        raise ValueError('Checkpoint lacks self-play run metadata')
    if saved.get('search_version') != SEARCH_VERSION:
        raise ValueError('Checkpoint uses legacy/unknown search training data. '
                         'Start model-v3 with a new --output and without --resume.')


def write_status(directory, status, settings, starting_game):
    payload = {'status': status, 'starting_game': starting_game, 'settings': settings,
               'before': str(directory / 'before.pt'), 'after': str(directory / 'after.pt'),
               'report': str(directory / 'evaluation.json')}
    temporary = directory / 'session.tmp'
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    temporary.replace(directory / 'session.json')


def main():
    parser = argparse.ArgumentParser(description='Real local Yut self-play; explicit assumptions in SELF_PLAY.md')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS, help='Default 4 parallel CPU games; MPS requires 1')
    parser.add_argument('--games', type=int, default=10, help='Additional games this invocation')
    parser.add_argument('--simulations', type=int, default=16)
    parser.add_argument('--difficulty', choices=list(DEPTH_LIMITS), default='expert')
    parser.add_argument('--rollout-steps', type=int, default=64)
    parser.add_argument('--max-actions', type=int, default=2000)
    parser.add_argument('--updates', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--replay-mib', type=int, default=DEFAULT_REPLAY_MIB)
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS, help='Learner PyTorch threads; rollout workers each use one thread')
    parser.add_argument('--device', choices=['cpu', 'mps', 'auto'], default=DEFAULT_DEVICE)
    parser.add_argument('--inference-backend', choices=['torch', 'coreml-ne'], default=DEFAULT_INFERENCE_BACKEND, help='Default: torch CPU inference')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/selfplay.pt'))
    parser.add_argument('--resume', type=Path)
    eval_counts = parser.add_mutually_exclusive_group()
    eval_counts.add_argument('--eval-games', type=int, help='Independent evaluation games after training; default 20')
    eval_counts.add_argument('--eval-pairs', type=int, help='Legacy count alias: N means 2*N independent games')
    args = parser.parse_args()
    args.eval_games = (args.eval_games if args.eval_games is not None else
                       2*args.eval_pairs if args.eval_pairs is not None else 20)
    for name in ('games', 'simulations', 'rollout_steps', 'max_actions', 'updates', 'batch_size', 'replay_mib', 'threads', 'eval_games', 'workers'):
        if getattr(args, name) < 1: parser.error(name+' must be positive')
    if args.resume and args.output.exists() and args.output.resolve() != args.resume.resolve():
        parser.error('Different output already exists; choose a new output')
    if args.output.exists() and args.resume is None:
        parser.error('Output exists; use --resume or a new --output')
    if args.inference_backend == 'coreml-ne':
        from .coreml_backend import require_coreml
        require_coreml()
    configure(args.threads)
    torch.manual_seed(args.seed)
    device = device_for(args.device)
    try:
        validate_inference_device(device, args.workers, args.inference_backend)
    except ValueError as exc:
        parser.error(str(exc))
    trainer = (Trainer.load(args.resume, device, schema=SCHEMA) if args.resume else
               Trainer(PolicyValueNet(ModelConfig(STATE_DIM, ACTION_DIM)), device, schema=SCHEMA))
    replay = Replay(trainer.model.config, args.replay_mib*1024**2, args.seed)
    completed, seed = 0, args.seed
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        try:
            validate_resume(saved)
        except ValueError as exc:
            parser.error(str(exc))
        run = saved['run']
        completed, seed = run['completed'], run['seed']
        for item in run['examples']: replay.add(Example(*item))
        replay.rng.setstate(run['replay_rng'])
        del saved, run
        if replay.items:
            del item
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    history = None
    if args.resume:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
        history = args.output.parent / (args.output.stem + '.history') / stamp
        history.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(args.resume, history / 'before.pt')
        write_status(history, 'training', settings, completed)
        print(json.dumps({'previous_checkpoint': str(history / 'before.pt')}), flush=True)
    log_path = args.output.with_suffix('.jsonl')
    print(json.dumps({'rules': RULESET, 'device': str(device), 'state_dim': STATE_DIM,
                      'threads': args.threads, 'workers': args.workers, 'batch_size': args.batch_size, 'replay_mib': args.replay_mib,
                      'search': SEARCH_VERSION, 'inference_backend': args.inference_backend,
                      'note': 'Training budget is experimental, not certified difficulty strength.'}), flush=True)
    workers = GameWorkers(args.workers, args.inference_backend)
    interrupted = False
    try:
        for round_start in range(completed, completed+args.games, args.workers):
            indices = list(range(round_start, min(round_start+args.workers, completed+args.games)))
            started = time.perf_counter()
            arguments = [(seed+i, args.simulations, DEPTH_LIMITS[args.difficulty],
                          args.max_actions, args.rollout_steps) for i in indices]
            results = workers.generate(trainer.model, arguments)
            generation_seconds = time.perf_counter()-started
            train_start = time.perf_counter()
            rows = []
            for game_index, ((game, examples, events), worker_seconds) in zip(indices, results):
                for example in examples: replay.add(example)
                losses = None
                if examples:
                    for _ in range(args.updates):
                        losses = trainer.train_batch(replay.sample(args.batch_size))
                rows.append({'game': game_index, 'seed': seed+game_index, 'winner': game.winner,
                   'status': 'completed' if game.winner is not None else 'truncated_no_targets',
                   'turns': game.turns, 'actions': len(events), 'examples': len(examples),
                   'worker_seconds': worker_seconds, 'loss': losses,
                   'rules': RULESET, 'search_version': SEARCH_VERSION, 'settings': settings, 'initial_seed': seed+game_index,
                   'generation_round': round_start, 'events': events})
            training_seconds = time.perf_counter()-train_start
            save_start = time.perf_counter()
            # Save only complete rounds: interrupted rounds are regenerated from
            # the last frozen-model boundary, without mixing pending stale games.
            save_run(trainer, replay, args.output, indices[-1]+1, seed, settings)
            save_seconds = time.perf_counter()-save_start
            with log_path.open('a') as stream:
                for row in rows:
                    stream.write(json.dumps(row)+'\n')
                    print(json.dumps({k: v for k, v in row.items() if k not in ('events','settings')}), flush=True)
            print(json.dumps({'round_end': indices[-1]+1, 'workers': args.workers,
                              'generation_seconds': generation_seconds,
                              'training_seconds': training_seconds, 'save_seconds': save_seconds,
                              'round_seconds': time.perf_counter()-started}), flush=True)
    except KeyboardInterrupt:
        interrupted = True
        if history:
            write_status(history, 'training_interrupted', settings, completed)
        print('Interrupted. Resume from the last saved round; unfinished round discarded.', flush=True)
        return
    finally:
        workers.close(interrupted=interrupted)
    if history:
        shutil.copyfile(args.output, history / 'after.pt')
        write_status(history, 'evaluating', settings, completed)
        command = [sys.executable, '-m', 'yut_ai.evaluate', str(history / 'after.pt'),
                   '--opponent-checkpoint', str(history / 'before.pt'),
                   '--games', str(args.eval_games), '--simulations', str(args.simulations),
                   '--rollout-steps', str(args.rollout_steps), '--threads', str(args.threads),
                   '--device', args.device, '--inference-backend', args.inference_backend, '--workers', str(args.workers), '--report', str(history / 'evaluation.json')]
        try:
            subprocess.run(command, check=True)
        except KeyboardInterrupt:
            write_status(history, 'evaluation_interrupted', settings, completed)
            print('Evaluation interrupted; trained checkpoint and both versions are preserved.', flush=True)
        except subprocess.CalledProcessError:
            write_status(history, 'evaluation_failed', settings, completed)
            raise
        else:
            write_status(history, 'complete', settings, completed)

if __name__ == '__main__':
    main()
