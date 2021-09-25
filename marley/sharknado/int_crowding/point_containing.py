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
from .. import int_crowding


class NoPoints(Exception):
    pass

class PointContainer(abc.ABC, collections.abc.Iterable, collections.abc.Container):

    intervals: tuple[int_crowding.Interval]

    def _arithmetic(self, other: ArithmeticOther, operator: Callable) -> IntCrowd:
        from .functions import intervals_to_points
        intervals = self._arithmetic_other_to_intervals(other)
        return IntCrowd(
            operator(set(intervals_to_points(self.intervals)), set(intervals_to_points(intervals))),
        )

    def __or__(self, other: ArithmeticOther) -> IntCrowd:
        intervals = self._arithmetic_other_to_intervals(other)
        return IntCrowd((self,) + intervals)

    def __and__(self, other: ArithmeticOther) -> IntCrowd:
        return self._arithmetic(other, operator_module.and_)

    def __xor__(self, other: ArithmeticOther) -> IntCrowd:
        return self._arithmetic(other, operator_module.xor)

    def __sub__(self, other: ArithmeticOther) -> IntCrowd:
        return self._arithmetic(other, operator_module.sub)

    def __contains__(self, other: ArithmeticOther) -> bool:
        from .functions import intervals_to_points
        intervals = self._arithmetic_other_to_intervals(other)
        return set(intervals_to_points(self.intervals)) >= set(intervals_to_points(intervals))

    def __bool__(self) -> bool:
        try:
            next(iter(self))
        except StopIteration:
            return False
        else:
            return True

    @abc.abstractmethod
    def __iter__(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def _reduce(self) -> tuple:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        return (type(other) == type(self)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())

    @staticmethod
    def _arithmetic_other_to_intervals(arithmetic_other: ArithmeticOther) -> \
                                                                       tuple[int_crowding.Interval]:
        from .int_crowding import IntCrowd
        if isinstance(arithmetic_other, PointContainer):
            return arithmetic_other.intervals
        elif isinstance(arithmetic_other, Point):
            return (arithmetic_other.as_interval(),)
        elif isinstance(arithmetic_other, Iterable):
            intervals = []
            for item in arithmetic_other:
                if isinstance(item, PointContainer):
                    intervals.extend(item.intervals)
                else:
                    assert isinstance(item, Point)
                    intervals.append(item.as_interval())
            return tuple(intervals)

    def get_last_point(self):
        if not self:
            raise NoPoints
        last_interval = self.intervals[-1]
        return Point((item if isinstance(item, int) else item[1] - 1)
                     for item in last_interval.as_tuple())

    def trim_inclusive(self) -> IntCrowd:
        if self:
            assert self.dimensions >= 1
        return IntCrowd(int_crowding.Interval(interval.as_tuple()[:-1])
                        for interval in self.intervals)

    def get_single_point(self) -> Point:
        (point,) = self
        return point



ArithmeticOther = Union[PointContainer, Point, Iterable[Union[PointContainer, Point]], Any]

from .int_crowding import IntCrowd
