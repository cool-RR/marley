# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional, Any
import itertools
import collections.abc

from ...jamming import BaseJamDatabase, Jam, JamId, JamItem, JamKind, JamParchment
from ..core import Swank, SwankDatabase, SwankRef
from .base_field import BaseField
from .. import utils


class SwankField(BaseField):

    field_type_name = 'swank'

    def from_jam(self, jam: Jam, swank_database: SwankDatabase, *,
                 swank_cache: dict) -> Optional[Swank]:
        if jam is None:
            return None
        else:
            jam_kind_name, jam_id_name, jam_index = jam
            return SwankRef(swank_database, utils.name_to_type(jam_kind_name),
                           JamId(jam_id_name), jam_index)

    def to_jam(self, value: Swank, swank_database: SwankDatabase) -> Jam:
        if value is None:
            return None
        else:
            assert value.has_jam_id_and_index
            swank_ref = SwankRef.from_swank_or_ref(value)
            return (swank_ref.jam_kind_name, str(swank_ref.jam_id), swank_ref.jam_index)

    def get_default_value(self, swank_database: SwankDatabase) -> None:
        return None

    def __get__(self, swank: Optional[Swank], swank_type: Optional[Type] = None) -> Optional[Swank]:
        if swank is None:
            assert swank_type is not None
            return self
        swank_ref: Optional[SwankRef] = swank._Swank__field_values[self.name]
        if swank_ref is None:
            return None
        else:
            return swank_ref.get()

    def __set__(self, swank: Swank,
                other_swank_or_ref: Optional[Union[Swank, SwankRef]]) -> None:
        if other_swank_or_ref is None:
            swank._Swank__field_values[self.name] = None
        else:
            swank._Swank__field_values[self.name] = \
                                             SwankRef.from_swank_or_ref(other_swank_or_ref)
