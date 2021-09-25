# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import operator as operator_module
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import collections.abc
import itertools


from .pointing import Point
from .point_containing import PointContainer, ArithmeticOther
from .functions import canonicalize_intervals, intervals_to_points, points_to_intervals


class IntCrowd(PointContainer):

    def __init__(self, stuff: ArithmeticOther = ()) -> None:
        self.intervals = canonicalize_intervals(
            self._arithmetic_other_to_intervals(stuff)
        )
        all_dimensions = {interval.dimensions for interval in self.intervals}
        if not all_dimensions:
            self.dimensions = None
        else:
            (self.dimensions,) = all_dimensions

    def __iter__(self) -> Iterable[Point]:
        return itertools.chain.from_iterable(self.intervals)

    def _reduce(self) -> tuple:
        return (type(self), self.intervals)

    def __repr__(self) -> str:
        if not self:
            intervals_str = ''
        elif len(self.intervals) == 1:
            intervals_str = repr(self.intervals[0])
        else:
            assert len(self.intervals) >= 2
            intervals_str = repr(self.intervals)

        return f'{type(self).__name__}({intervals_str})'

    def to_savvy_content(self):
        return (self.intervals,)

    def separate(self, axes: Tuple[int]) -> Tuple[IntCrowd]:
        if not axes:
            return (self,)
        elif len(axes) > 1:
            raise NotImplementedError
        return tuple(IntCrowd(points) for (_, points) in
                     itertools.groupby(self, lambda point: tuple(point[axis] for axis in axes)))

