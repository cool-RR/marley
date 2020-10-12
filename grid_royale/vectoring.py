# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import dataclasses
import itertools
import operator
import functools
import math
from typing import (Optional, Tuple, Union, Container, Hashable, Iterator,
                    Iterable, Any, Dict, FrozenSet)
import numpy as np


from . import base


@dataclasses.dataclass(frozen=True)
class Vector:
    '''A vector in 2 dimensions, e.g. Vector(-2, 3)'''

    x: int
    y: int

    __iter__ = lambda self: iter((self.x, self.y))

    def __matmul__(self, other: Vector) -> int:
        '''Get Levenshtein distance between two vectors'''
        return sum(map(abs, map(operator.sub, self, other)))

    def __neg__(self) -> Vector:
        return type(self)(x=-self.x, y=-self.y)

    __bool__ = lambda self: any(self)

    def rotate_in_board(self, n_right_angles: int, /, board_size: int) -> Vector:
        x, y = self
        for _ in range(n_right_angles % 4):
            (x, y) = (board_size - 1 - y, x)
        return type(self)(x, y)

    def iterate_rotations_in_board(self, /, board_size: int) -> Iterator[Vector]:
        x, y = self
        yield self
        for _ in range(3):
            (x, y) = (board_size - 1 - y, x)
            yield type(self)(x, y)

    def __hash__(self):
        return hash((type(self), *self))

    def __eq__(self, other: Any):
        return (type(self) is type(other)) and (tuple(self) == tuple(other))




class Position(Vector):
    @staticmethod
    def iterate_all(state_or_board_size: Union[base._BaseGrid, int], /):
        board_size: int = (state_or_board_size if isinstance(state_or_board_size, int)
                           else state_or_board_size.board_size)
        for y, x in itertools.product(range(board_size), repeat=2):
            yield Position(x, y)

    def __sub__(self, other: Union[Position, Translation]):
        if isinstance(other, Position):
            result_type = Translation
        else:
            assert isinstance(other, Translation)
            result_type = Position

        return result_type(self.x - other.x, self.y - other.y)


    def __add__(self, translation: Translation):
        assert isinstance(translation, Translation)
        return Position(self.x + translation.x,
                        self.y + translation.y)


    def in_square(self, square_size: int) -> bool:
        return ((0 <= self.x <= square_size - 1) and
                (0 <= self.y <= square_size - 1))

    def walk(self, translation: Translation,
             board_size: Optional[int] = None) -> Iterator[Position]:
        position = self
        if board_size is None:
            while True:
                yield position
                position += translation
        else:
            while 0 <= min(position) <= max(position) <= board_size - 1:
                yield position
                position += translation


    @functools.lru_cache(maxsize=None)
    def field_of_view(self, vicinity: Vicinity, board_size: int) -> Tuple[FrozenSet[Position]]:
        result = []
        count_from_one = itertools.count(start=1)
        if vicinity._is_step_like():
            for i in count_from_one:
                positions = frozenset(
                    position for j in range(-i, i + 1) if (position := (self +
                    i * vicinity + j * Step(*vicinity).perpendiculars[0])).in_square(board_size)
                )
                if not positions:
                    return tuple(result)
                result.append(positions)
        else:
            for i in count_from_one:
                positions = frozenset(
                    position for j in range(i + 1) if (position := (
                    self + Translation(j * vicinity.x, (i - j) * vicinity.y))).in_square(board_size)
                )
                if not positions:
                    return tuple(result)
                result.append(positions)


    def horizontal_line_in_board(self, board_size: int) -> Tuple[Position]:
        return tuple(Position(i, self.y) for i in range(board_size))

    def vertical_line_in_board(self, board_size: int) -> Tuple[Position]:
        return tuple(Position(self.x, i) for i in range(board_size))








PositionLike = Union[Position, Tuple[int, int]]

class Translation(Vector):
'''A Translation (i.e. movement) in 2-dimensional space.'''
    def __mul__(self, number: int) -> Translation:
        assert isinstance(number, int)
        return type(self)(x=(self.x * number), y=(self.y * number))

    def __rmul__(self, number: int) -> Translation:
        return self * number

    def _is_step_like(self):
        return tuple(self).count(0) == 1


class Vicinity(Translation):
    all_vicinities: Tuple[Vicinity]
    def __init__(self, x: int, y: int):
        assert {0} != set((x, y)) <= {-1, 0, 1}
        super().__init__(x=x, y=y)


    def __mul__(self, number: int) -> Translation:
        assert isinstance(number, int)
        return Translation(x=(self.x * number), y=(self.y * number))

Vicinity.all_vicinities = tuple(
    itertools.starmap(
        Vicinity,
        filter(
            any,
            itertools.product((-1, 0, 1), repeat=2)
        )
    )
)


class Step(Vicinity):

    _all_ascii = '↑→↓←'

    def __init__(self, x: int, y: int):
        assert frozenset((x, y)) in {frozenset((0, 1)),
                                     frozenset((0, -1))}
        super().__init__(x=x, y=y)

    @property
    def index(self):
        try:
            return self._index
        except AttributeError:
            self._index = tuple(Step.all_steps).index(self)
            return self._index

    @property
    def name(self) -> str:
        try:
            return self._name
        except AttributeError:
            self._name = ('up', 'right', 'down', 'left')[self.index]
            return self._name

    @property
    def ascii(self) -> str:
        try:
            return self._ascii
        except AttributeError:
            self._ascii = Step._all_ascii[tuple(Step.all_steps).index(self)]
            return self._ascii

    def is_general_direction(self, source: Position, target: Position) -> bool:
        translation: Translation = target - source
        if self.x:
            return ((np.sign(translation.x) == self.x) and
                    abs(translation.x) >= abs(translation.y))
        else:
            assert self.y
            return ((np.sign(translation.y) == self.y) and
                    abs(translation.y) >= abs(translation.x))


    @property
    def perpendiculars(self) -> Tuple[Step]:
        try:
            return self._perpendiculars
        except AttributeError:
            self._perpendiculars = tuple(
                sorted(
                    ((first := type(self)(*reversed(tuple(self)))), -first),
                    key=tuple
                )
            )
            return self._perpendiculars

    @property
    def angle_from_top(self):
        return math.tau * (self.index / 4)


(Step.up, Step.right, Step.down, Step.left) = Step.all_steps = (
    Step(0, -1), Step(1, 0), Step(0, 1), Step(-1, 0)
)
