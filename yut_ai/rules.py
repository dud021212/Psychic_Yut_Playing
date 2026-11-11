"""Confirmed rule primitives. Not a complete board/game engine."""
from dataclasses import dataclass
from enum import IntEnum

RULES_VERSION = 'local-v2-hidden-powers'

class Power(IntEnum):
    DOG_PIG = 0
    HORSEPOWER = 1
    REVERSE_GEAR = 2
    DOUBLE = 3
    ALL_OR_NOTHING = 4
    MIDAS = 5
    TAKEOUT = 6
    BACKSTEP = 7

NORMAL = {'do': 3/16, 'gae': 6/16, 'geol': 4/16, 'yut': 1/16,
          'mo': 1/16, 'backdo': 1/16}
DISTANCE = {'do': 1, 'gae': 2, 'geol': 3, 'yut': 4, 'mo': 5, 'backdo': -1}
SPECIAL = {
    Power.DOG_PIG: {'do_and_gae': 1.0},
    Power.HORSEPOWER: {'geol': 1.0},
    Power.REVERSE_GEAR: {'backdo': 1.0},
    Power.ALL_OR_NOTHING: {'do': 0.5, 'mo': 0.5},
    Power.MIDAS: {'yut': 0.5, 'mo': 0.5},
}

@dataclass(frozen=True)
class MoveResult:
    source: str
    distance: int
    backstep: bool = False

@dataclass(frozen=True)
class ThrowResult:
    moves: tuple[MoveResult, ...]
    grants_extra: bool
    nak: bool

def throw_distribution(power=None, *, first_throw=True):
    if power is not None:
        power = Power(power)
    if first_throw and power in SPECIAL:
        return dict(SPECIAL[power])
    return {'nak': 0.015, **{key: 0.985 * p for key, p in NORMAL.items()}}

def resolve_throw(outcome, power=None, *, first_throw=True):
    if outcome not in throw_distribution(power, first_throw=first_throw):
        raise ValueError('Outcome impossible under this throw distribution')
    active = Power(power) if first_throw and power is not None else None
    if outcome == 'nak':
        return ThrowResult((), False, True)
    if outcome == 'do_and_gae':
        return ThrowResult((MoveResult('do', 1), MoveResult('gae', 2)), False, False)
    backstep = active == Power.BACKSTEP
    distance = -DISTANCE[outcome] if backstep else DISTANCE[outcome]
    move = MoveResult(outcome, distance, backstep)
    return ThrowResult((move,) * (2 if active == Power.DOUBLE else 1),
                       outcome in ('yut', 'mo'), False)

def extra_after_capture(current: bool, source: str) -> bool:
    if source not in DISTANCE:
        raise ValueError(source)
    return current or source not in ('yut', 'mo')

def discard_unusable_backdo(*, on_board: bool, has_forward: bool, extra: bool) -> bool:
    return not on_board and not has_forward and not extra

def resolve_path(path, *, finish, start, move, gate=None, bomb=None):
    """Resolve an already validated path; returns destination and event.

    Route graph and occupancy/capture are caller responsibilities. Path must
    include each traversed cell in order, not merely the final destination.
    """
    if gate is not None and gate == bomb:
        raise ValueError('Installations cannot overlap')
    if finish is None or start is None:
        raise ValueError("Path resolution requires board cells, not off-board sentinels")
    if gate == finish or bomb == finish:
        raise ValueError('No installations on finish')
    destination = start
    for cell in path:
        if cell is None:
            raise ValueError("Completion/off-board transition belongs to the game engine")
        destination = cell
        if cell == gate:
            return finish, 'gate'
        if cell == bomb:
            return None, 'bomb'
        if move.backstep and move.distance < 0 and start != finish and cell == finish:
            return finish, 'backstep_stop'
    return destination, 'normal'

def shuffle_groups(group_ids, allowed_cells, rng):
    """Uniform injective placement is a configurable implementation assumption."""
    groups, cells = tuple(group_ids), tuple(allowed_cells)
    if len(set(groups)) != len(groups) or len(set(cells)) != len(cells):
        raise ValueError('Duplicate group/cell')
    if len(groups) > len(cells):
        raise ValueError('Insufficient cells')
    return dict(zip(groups, rng.sample(cells, len(groups))))
