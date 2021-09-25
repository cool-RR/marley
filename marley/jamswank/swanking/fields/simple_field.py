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
from .base_field import BaseField


class SimpleField(BaseField):
    def __init__(self, default_factory: Optional[Callable[[], Any]] = type(None)) -> None:
        self.default_factory = default_factory

    def from_jam(self, jam: Jam, swank_database: SwankDatabase, *, swank_cache: dict) -> Swank:
        return jam

    def to_jam(self, value: Optional[Any], swank_database: SwankDatabase) -> Jam:
        return value

