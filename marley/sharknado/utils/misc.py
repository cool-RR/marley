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

_initial_missing = object()


def union(items: Iterable[Any], initial: Any = _initial_missing) -> Any:
    extra_args = (initial,) if initial is not _initial_missing else ()
    return functools.reduce(operator_module.or_, items, *extra_args)

def intersection(items: Iterable[Any], initial: Any = _initial_missing) -> Any:
    extra_args = (initial,) if initial is not _initial_missing else ()
    return functools.reduce(operator_module.and_, items, *extra_args)


class ChangeTracker:
    def __init__(self):
        self.value = object()

    def __call__(self, value: Any) -> bool:
        try:
            return (self.value != value)
        finally:
            self.value = value


