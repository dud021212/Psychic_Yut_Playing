"""Playable local rules. Explicit assumptions are documented in SELF_PLAY.md."""
from dataclasses import dataclass, field
import copy
from functools import lru_cache
import random
from .rules import Power, MoveResult, throw_distribution, resolve_throw, extra_after_capture
from .belief import PowerBelief

HOME, DONE = -1, 29
RULESET = 'local-play-v1'

def backward(cell):
    branches = {0: (19, 28), 15: (14, 24), 22: (21, 26)}
    inner = {20: 5, 21: 20, 23: 22, 24: 23, 25: 10, 26: 25, 27: 22, 28: 27}
    return branches.get(cell, (inner.get(cell, cell - 1),))

def paths(start, move):
    return list(_paths(start, move))

@lru_cache(maxsize=2048)
def _paths(start, move):
    if start == DONE or (start == HOME and move.distance < 0):
        return ()
    def walk(cell, remaining, first, path):
        if not remaining or cell == DONE:
            return [tuple(path)]
        if move.distance > 0:
            inner = {20: 21, 21: 22, 22: 23, 23: 24, 24: 15,
                     25: 26, 26: 22, 27: 28, 28: 0}
            if cell == HOME:
                options = (1,)
            elif cell == 0:
                options = (DONE,)
            elif cell == 22 and start in (10, 25, 26):
                # Continue straight along the upper-left to finish diagonal.
                options = (27,)
            elif first and cell in (5, 10, 22):
                options = ({5: 20, 10: 25, 22: 27}[cell],)
            else:
                options = (inner.get(cell, (cell + 1) % 20),)
        else:
            options = backward(cell)
        result = []
        for dest in options:
            trail = path + [dest]
            if move.backstep and move.distance < 0 and start != 0 and dest == 0:
                result.append(tuple(trail))
            else:
                result.extend(walk(dest, remaining - 1, False, trail))
        return result
    return tuple(walk(start, abs(move.distance), True, []))

@dataclass(frozen=True)
class Action:
    kind: str
    args: tuple = ()

@dataclass
class Game:
    hands: list
    positions: list = field(default_factory=lambda: [[HOME]*4, [HOME]*4])
    used: list = field(default_factory=lambda: [[False]*2, [False]*2])
    beliefs: list = field(default_factory=list)
    player: int = 0
    phase: str = 'power'
    moves: list = field(default_factory=list)
    stored: list = field(default_factory=lambda: [[], []])
    extra: bool = False
    active: int | None = None
    takeout: bool = False
    installations: dict = field(default_factory=dict)
    turns: int = 0
    winner: int | None = None

    @classmethod
    def new(cls, rng):
        a, b = rng.sample(range(8), 2), rng.sample(range(8), 2)
        hands = [[a[0], b[0]], [a[1], b[1]]]
        return cls(hands=hands, beliefs=[PowerBelief(*h) for h in hands])

    def clone(self):
        cloned = copy.copy(self)
        cloned.hands = [row.copy() for row in self.hands]
        cloned.positions = [row.copy() for row in self.positions]
        cloned.used = [row.copy() for row in self.used]
        cloned.moves = self.moves.copy()  # MoveResult is frozen.
        cloned.stored = [row.copy() for row in self.stored]
        cloned.installations = self.installations.copy()
        cloned.beliefs = []
        for belief in self.beliefs:
            item = copy.copy(belief)
            item.probabilities = belief.probabilities.copy()
            cloned.beliefs.append(item)
        return cloned

    def groups(self, player, *, board_only=False):
        seen, result = set(), []
        for i, pos in enumerate(self.positions[player]):
            if pos == DONE or (board_only and pos == HOME):
                continue
            if pos >= 0 and pos in seen:
                continue
            seen.add(pos)
            result.append(i)
        return result

    def members(self, player, index):
        pos = self.positions[player][index]
        return [index] if pos == HOME else [i for i, p in enumerate(self.positions[player]) if p == pos]

    def empty(self, cell):
        return cell not in self.installations and all(cell not in row for row in self.positions)

    def legal_actions(self):
        if self.winner is not None:
            return []
        p, q = self.player, 1-self.player
        if self.phase == 'throw':
            return [Action('throw')]
        if self.phase == 'power':
            actions = [Action('skip')]
            power = self.hands[p][0]
            if not self.used[p][0] and (power != Power.REVERSE_GEAR or self.groups(p, board_only=True)):
                actions.append(Action('power', (0,)))
            if self.used[p][1]:
                return actions
            power = self.hands[p][1]
            own, other = self.groups(p, board_only=True), self.groups(q, board_only=True)
            if power == 0:  # carry home piece onto own group
                home = next((i for i, x in enumerate(self.positions[p]) if x == HOME), None)
                if home is not None:
                    actions += [Action('power', (1, i, home)) for i in own]
            elif power == 1:
                actions += [Action('power', (1, j)) for j in other]
            elif power == 2:
                actions += [Action('power', (1, i, j)) for i in own for j in other]
            elif power == 3:
                for i in own:
                    cells = set()
                    for n in (1, 2, 3):
                        for path in paths(self.positions[p][i], MoveResult('do', n)):
                            if path[-1] != DONE and self.empty(path[-1]):
                                cells.add(path[-1])
                    actions += [Action('power', (1, i, j, c)) for j in other for c in sorted(cells)]
            elif power == 4:
                for i in own:
                    for j in other:
                        for cell in backward(self.positions[q][j]):
                            if cell not in self.positions[q]:
                                actions.append(Action('power', (1, i, j, cell)))
            elif power == 5:
                if own or other:
                    actions.append(Action('power', (1,)))
            else:
                actions += [Action('power', (1, c)) for c in range(1, 29) if self.empty(c)]
            return actions
        actions = [Action('throw')] if self.extra else []
        if not self.takeout:
            seen = set()
            for k, move in enumerate(self.moves):
                if move in seen:
                    continue
                seen.add(move)
                for i in self.groups(p):
                    for path in paths(self.positions[p][i], move):
                        actions.append(Action('move', (k, i, path)))
        return actions or [Action('end')]

    def relocate(self, player, members, destination):
        for i in members:
            self.positions[player][i] = destination

    def land(self, members, destination, source=None):
        p, q = self.player, 1-self.player
        if 0 <= destination < DONE:
            targets = [i for i, x in enumerate(self.positions[q]) if x == destination]
            if targets:
                if source is None:
                    raise ValueError('Direct power cannot capture')
                self.relocate(q, targets, HOME)
                self.extra = extra_after_capture(self.extra, source)
        self.relocate(p, members, destination)

    def apply(self, action, rng, *, validate=True):
        if validate and action not in self.legal_actions():
            raise ValueError(f'Illegal action: {action}')
        p, q = self.player, 1-self.player
        event = {'player': p, 'action': action.kind, 'args': action.args}
        if action.kind == 'skip':
            self.phase = 'throw'
        elif action.kind == 'power':
            category = action.args[0]
            power = self.hands[p][category]
            self.used[p][category] = True
            # Assumption: the used ability identity becomes publicly observable.
            self.beliefs[q].observe_used_power(category, power)
            event['power'] = (category, power)
            if category == 0:
                self.active = power
                self.takeout = power == Power.TAKEOUT
            else:
                args = action.args[1:]
                if power == 0:
                    i, home = args
                    self.positions[p][home] = self.positions[p][i]
                elif power == 1:
                    self.relocate(q, self.members(q, args[0]), HOME)
                elif power == 2:
                    i, j = args
                    own, other = self.members(p, i), self.members(q, j)
                    a, b = self.positions[p][i], self.positions[q][j]
                    self.relocate(p, own, b)
                    self.relocate(q, other, a)
                elif power == 3:
                    _, j, cell = args
                    self.relocate(q, self.members(q, j), cell)
                elif power == 4:
                    i, _, cell = args
                    members = self.members(p, i)
                    # Configured assumption: direct backhug triggers installations.
                    installation = self.installations.pop(cell, None)
                    if installation == 'gate':
                        # No throw result exists; no capture bonus is granted.
                        target = [j for j, x in enumerate(self.positions[q]) if x == 0]
                        self.relocate(q, target, HOME)
                        self.relocate(p, members, 0)
                    else:
                        self.land(members, HOME if installation == 'bomb' else cell)
                elif power == 5:
                    groups = [(owner, self.members(owner, i)) for owner in (0, 1)
                              for i in self.groups(owner, board_only=True)]
                    cells = rng.sample([c for c in range(29) if c not in self.installations], len(groups))
                    for (owner, members), cell in zip(groups, cells):
                        self.relocate(owner, members, cell)
                    event['shuffle'] = cells
                else:
                    self.installations[args[0]] = 'gate' if power == 6 else 'bomb'
            self.phase = 'throw'
        elif action.kind == 'throw':
            first = self.phase == 'throw'
            if not first:
                self.extra = False
            dist = throw_distribution(self.active, first_throw=first)
            outcome = rng.choices(list(dist), weights=list(dist.values()), k=1)[0]
            result = resolve_throw(outcome, self.active, first_throw=first)
            if first:
                self.moves.extend(self.stored[p])
                self.stored[p] = []
            self.moves.extend(result.moves)
            self.extra = self.extra or result.grants_extra
            self.phase = 'move'
            event['outcome'] = outcome
        elif action.kind == 'move':
            k, i, path = action.args
            move = self.moves.pop(k)
            members = self.members(p, i)
            destination = path[-1]
            for cell in path:
                installation = self.installations.pop(cell, None)
                if installation:
                    destination = 0 if installation == 'gate' else HOME
                    break
            self.land(members, destination, move.source)
        elif action.kind == 'end':
            if self.takeout:
                self.stored[p].extend(self.moves)
            self.moves = []
            self.extra = False
            self.active = None
            self.takeout = False
            self.player = q
            self.phase = 'power'
            self.turns += 1
        else:
            raise ValueError(action)
        for owner in (0, 1):
            if all(x == DONE for x in self.positions[owner]):
                self.winner = owner
        self.check()
        return event

    def check(self):
        assert all(len(row) == 4 and all(HOME <= x <= DONE for x in row) for row in self.positions)
        assert not (set(self.positions[0]) & set(self.positions[1]) - {HOME, DONE})
        assert all(1 <= c < DONE for c in self.installations)
        assert all(c not in self.positions[0] + self.positions[1] for c in self.installations)
