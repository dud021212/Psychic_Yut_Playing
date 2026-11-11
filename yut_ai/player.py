"""Observation-only network interface and root information-set rollout search.

This is a root bandit with neural-policy rollouts, NOT a full recursive IS-MCTS.
All root action statistics are shared across sampled opponent hands.
"""
import math
import random
import torch
from .game import HOME, DONE, Action
from .observation import PlayerObservation
from .rules import DISTANCE

SCHEMA = 'local-play-v1-public-belief-features-v1'
ACTION_DIM = 48
SEARCH_VERSION = 'root-information-set-bandit-neural-rollouts-v3'

def features(game):
    p, q = game.player, 1-game.player
    public = []
    for owner in (p, q):
        public += [game.positions[owner].count(c)/4 for c in range(HOME, DONE+1)]
    for c in range(29):
        public += [float(game.installations.get(c) == kind) for kind in ('gate', 'bomb')]
    public += [float(x) for owner in (p, q) for x in game.used[owner]]
    public += [float(game.phase == phase) for phase in ('power', 'throw', 'move')]
    public += [float(game.extra), float(game.takeout)]
    # Public moves retain their original type and direction.
    for moves in (game.moves, game.stored[p], game.stored[q]):
        for source in DISTANCE:
            for direction in (-1, 1):
                public.append(math.log1p(sum(m.source == source and m.distance*direction > 0 for m in moves)))
    public += [float(game.active == i) for i in range(8)]
    return PlayerObservation.from_belief(public, game.beliefs[p]).encode()

# Sum of public features + own one-hots + joint posterior.
STATE_DIM = 2*31 + 58 + 4 + 3 + 2 + 36 + 8 + 80

def action_features(game, action):
    values = [0.0]*ACTION_DIM
    values[('skip', 'power', 'throw', 'move', 'end').index(action.kind)] = 1
    p = game.player
    if action.kind == 'move':
        k, i, path = action.args
        move = game.moves[k]
        values[5 + list(DISTANCE).index(move.source)] = 1
        values[11] = move.distance / 5
        values[12] = len(game.members(p, i))/4
        destination = path[-1]
        for cell in path:
            if cell in game.installations:
                destination = 0 if game.installations[cell] == 'gate' else HOME
                break
        values[13] = (game.positions[p][i]+1)/30
        values[14] = (destination+1)/30
        values[15] = game.positions[1-p].count(destination)/4 if 0 <= destination < DONE else 0
        values[16] = game.positions[p].count(destination)/4 if 0 <= destination < DONE else 0
        for j, cell in enumerate(path[:5]):
            values[17+j] = (cell+1)/30
    elif action.kind == 'power':
        category = action.args[0]
        values[22 + category*8 + game.hands[p][category]] = 1
        for j, arg in enumerate(action.args[1:]):
            values[38+j] = (arg+1)/30
    return values

def progress(cell):
    if cell == HOME: return 0
    if cell == DONE: return 32
    if cell == 0: return 29
    return {20: 10, 21: 12, 22: 23, 23: 14, 24: 15, 25: 12,
            26: 14, 27: 25, 28: 27}.get(cell, cell)

def heuristic(game, action):
    if action.kind == 'move':
        _, i, path = action.args
        dest = path[-1]
        for c in path:
            if c in game.installations:
                dest = 0 if game.installations[c] == 'gate' else HOME
                break
        size = len(game.members(game.player, i))
        score = size*(progress(dest)-progress(game.positions[game.player][i]))
        if 0 <= dest < DONE:
            score += 12*game.positions[1-game.player].count(dest)
        return score
    if action.kind == 'power': return 1.5
    if action.kind == 'throw': return 2
    return 0

def rollout_action(game, rng):
    # This rule uses board, own hand, phase only, never opponent's true hand.
    actions = game.legal_actions()
    scores = [heuristic(game, a) for a in actions]
    best = max(scores)
    return rng.choices(actions, weights=[math.exp((s-best)/4) for s in scores], k=1)[0]

@torch.inference_mode()
def evaluate(model, game, actions):
    if hasattr(model, "evaluate_features"):
        return model.evaluate_features(features(game), [action_features(game, a) for a in actions])
    device = next(model.parameters()).device
    state = torch.tensor([features(game)], dtype=torch.float32, device=device)
    acts = torch.tensor([[action_features(game, a) for a in actions]], dtype=torch.float32, device=device)
    legal = torch.ones((1, len(actions)), dtype=torch.bool, device=device)
    probs, value = model.predict(state, acts, legal)
    return probs[0].cpu().tolist(), float(value[0])

def neural_rollout_action(game, model, rng):
    """Both simulated players use this frozen policy from their own observation.

    Forced actions need no choice. Otherwise sample the policy rather than
    recursing into another search or consulting the fixed heuristic opponent.
    """
    actions = game.legal_actions()
    if len(actions) == 1:
        return actions[0]
    policy, _ = evaluate(model, game, actions)
    return rng.choices(actions, weights=policy, k=1)[0]


def search(game, model, rng, *, simulations=32, depth_turns=4, rollout_steps=128):
    actions = game.legal_actions()
    if len(actions) == 1:
        return actions, [1.0], {'simulations': 0, 'truncated_rollouts': 0}
    priors, _ = evaluate(model, game, actions)
    # Exact immediate wins are never missed merely because of rollout budget.
    for i, action in enumerate(actions):
        if action.kind == 'move':
            trial = game.clone()
            trial.apply(action, random.Random(0), validate=False)
            if trial.winner == game.player:
                return actions, [float(j == i) for j in range(len(actions))], {'simulations': 0, 'truncated_rollouts': 0}
    visits, totals = [0]*len(actions), [0.0]*len(actions)
    truncated = 0
    root = game.player
    # All legal root actions receive at least one sample.
    budget = max(simulations, 4 * len(actions))
    for iteration in range(budget):
        if iteration < len(actions):
            index = iteration
        else:
            index = max(range(len(actions)), key=lambda i: totals[i]/visits[i] +
                        1.5*priors[i]*math.sqrt(iteration)/(1+visits[i]))
        trial = game.clone()
        trial.hands[1-root] = list(game.beliefs[root].sample(rng))
        # Reconstruct opponent inference from its sampled own hand and publicly
        # revealed root powers. Do not inherit an impossible private posterior.
        from .belief import PowerBelief
        trial.beliefs[1-root] = PowerBelief(*trial.hands[1-root])
        for category in (0, 1):
            if game.used[root][category]:
                trial.beliefs[1-root].observe_used_power(category, game.hands[root][category])
        trial.apply(actions[index], rng, validate=False)
        count = 0
        while trial.winner is None and (depth_turns is None or trial.turns-game.turns < depth_turns) and count < rollout_steps:
            trial.apply(neural_rollout_action(trial, model, rng), rng, validate=False)
            count += 1
        if trial.winner is not None:
            value = 1.0 if trial.winner == root else -1.0
        else:
            truncated += int(count >= rollout_steps)
            legal = trial.legal_actions()
            _, value = evaluate(model, trial, legal)
            value *= 1 if trial.player == root else -1
        visits[index] += 1
        totals[index] += value
    return actions, [n/budget for n in visits], {'simulations': budget, 'truncated_rollouts': truncated}
