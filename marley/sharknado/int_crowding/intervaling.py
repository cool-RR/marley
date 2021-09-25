# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import operator as operator_module
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import collections.abc
import itertools
import functools

from .pointing import Point
from .point_containing import PointContainer, ArithmeticOther


class IntervalType(type(PointContainer)):
    def __getitem__(cls, arg) -> Interval:
        as_tuple = arg if isinstance(arg, tuple) else (arg,)
        return Interval(as_tuple)

def slice_to_range(s: slice) -> range:
    assert s.step is None
    return range(s.start, s.stop)


@functools.total_ordering
class Interval(PointContainer, metaclass=IntervalType):

    def __init__(self, numbers_and_slices: Iterable[Union[int, tuple[int, int]]]) -> None:
        processed_numbers_and_slices = []
        numbers_and_slices_tuple = tuple(numbers_and_slices)
        for item in numbers_and_slices_tuple:
            if isinstance(item, int):
                assert item >= 0
                processed_numbers_and_slices.append(item)
            else:
                if isinstance(item, tuple):
                    (start, end) = item
                else:
                    assert isinstance(item, slice)
                    start = item.start
                    end = item.stop

                assert isinstance(start, int)
                assert isinstance(end, int)
                assert 0 <= start <= end
                if start <= end - 2:
                    processed_numbers_and_slices.append(slice(start, end))
                elif start == end - 1:
                    processed_numbers_and_slices.append(start)
                else:
                    assert start == end
                    processed_numbers_and_slices = [slice(0, 0)] * len(numbers_and_slices_tuple)
                    break

        self.numbers_and_slices = tuple(processed_numbers_and_slices)
        self.dimensions = len(self.numbers_and_slices)
        self.sliceful_tuple = tuple((slice(item, item + 1) if isinstance(item, int) else item)
                                    for item in self.numbers_and_slices)
        self.tupleful_tuple = tuple((s.start, s.stop) for s in self.sliceful_tuple)
        self.intervals = (self,)

    def as_tuple(self) -> tuple:
        return self.numbers_and_slices

    def __iter__(self) -> Iterable[Point]:
        return map(Point, itertools.product(*map(slice_to_range, self.sliceful_tuple)))

    def _reduce(self) -> tuple:
        return (type(self), repr(self))

    def __repr__(self) -> str:
        if self.dimensions == 0:
            assert self.numbers_and_slices == ()
            return f'{type(self).__name__}(())'
        else:
            foo = ', '.join(
                str(item) if isinstance(item, int) else f'{item.start}:{item.stop}'
                for item in self.numbers_and_slices
            )
            return f'{type(self).__name__}[{foo}]'

    def __lt__(self, other: Interval) -> bool:
        assert isinstance(other, Interval)
        if self.dimensions != other.dimensions:
            return (self.dimensions < other.dimensions)
        else:
            return (self.sliceful_tuple < other.sliceful_tuple)

    def to_savvy_content(self):
        return (self.tupleful_tuple,)



from .int_crowding import IntCrowd
from .functions import intervals_to_points, points_to_intervals, canonicalize_intervals