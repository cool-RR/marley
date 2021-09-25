# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional, Any
import itertools
import collections.abc

from ...jamming import BaseJamDatabase, Jam, JamId, JamItem, JamKind, JamParchment
from ..core import Swank, SwankDatabase



class BaseField(abc.ABC):
    @abc.abstractmethod
    def from_jam(self, jam: Jam, swank_database: SwankDatabase, *, swank_cache: dict) -> Swank:
        raise NotImplementedError

    @abc.abstractmethod
    def to_jam(self, value: Optional[Any], swank_database: SwankDatabase) -> Jam:
        raise NotImplementedError

    def get_default_value(self, swank_database: SwankDatabase) -> Any:
        return self.default_factory()

    default_factory = type(None)

    def __set_name__(self, swank_type: Type[Swank], name) -> None:
        self.name = name

    def __get__(self, swank: Optional[Swank], swank_type: Optional[Type] = None) -> Any:
        if swank is None:
            assert swank_type is not None
            return self
        return swank._Swank__field_values[self.name]

    def __set__(self, swank: Swank, value: Any) -> None:
        swank._Swank__field_values[self.name] = value

    def __delete__(self, swank: Swank) -> None:
        raise NotImplementedError

    def _reduce(self) -> tuple:
        return (type(self), self.name)

    def __eq__(self, other: Any) -> bool:
        return (type(other) == type(self)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.name)})'




