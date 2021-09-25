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


@functools.total_ordering
class Point(collections.abc.Sequence):
    def __init__(self, numbers: Iterable[int]) -> None:
        self.numbers = tuple(numbers)
        for number in self.numbers:
            assert isinstance(number, int)
            assert number >= 0
        self.dimensions = len(self.numbers)

    def __getitem__(self, i: int) -> int:
        return self.numbers[i]

    def __len__(self) -> int:
        return len(self.numbers)

    def __iter__(self) -> Iterable[int]:
        return iter(self.numbers)

    def __reversed__(self) -> Iterable[int]:
        return reversed(self.numbers)

    def as_tuple(self) -> tuple:
        return self.numbers

    def as_interval(self) -> Interval:
        return Interval(self)

    def __lt__(self, other: Point) -> bool:
        assert isinstance(other, Point)
        if self.dimensions != other.dimensions:
            return (self.dimensions < other.dimensions)
        else:
            return (self.numbers < other.numbers)

    def __eq__(self, other: Point) -> bool:
        assert isinstance(other, Point)
        return self.numbers == other.numbers

    def __hash__(self) -> int:
        return hash((type(self), self.numbers))

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.numbers})'



from .intervaling import Interval