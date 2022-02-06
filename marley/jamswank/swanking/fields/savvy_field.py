# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional, Any, Sequence, Callable
import itertools
import collections.abc
import functools
import enum
import numbers
import re
import base64
import dataclasses

import numpy as np

from ...jamming import BaseJamDatabase, Jam, JamId, JamItem, JamKind, JamParchment
from marley.utils import ImmutableDict
from ..core import Swank, SwankDatabase
from .base_field import BaseField
from .. import utils



@functools.cache
def _dtype_to_savvy_content(dtype: np.dtype) -> tuple:
    savvy_content = (eval(re.fullmatch(f'dtype\((.+)\)', repr(dtype)).group(1)),) # God forgive me.
    assert np.dtype(*savvy_content) == dtype
    return savvy_content


def object_to_savvy_content(o: Any) -> tuple:
    if hasattr(o, 'to_savvy_content'):
        return o.to_savvy_content()
    elif isinstance(o, (tuple, set, frozenset)):
        return (list(o),)
    elif isinstance(o, (dict, ImmutableDict)):
        return (list(map(list, o.items())),)
    elif dataclasses.is_dataclass(type(o)):
        return tuple(getattr(o, field_name) for field_name in o.__dataclass_fields__)
    elif isinstance(o, np.dtype):
        return _dtype_to_savvy_content(o)
    elif isinstance(o, bytes):
        return (base64.b64encode(o).decode(),)
    elif isinstance(o, type):
        return (utils.type_to_name(o),)
    else:
        assert isinstance(o, enum.Enum)
        return (o.value,)


class SavvyField(BaseField):

    field_type_name = 'savvy'

    def __init__(self, default_factory: Optional[Callable[[], Any]] = type(None)) -> None:
        self.default_factory = default_factory


    def from_jam(self, jam: Jam, swank_database: SwankDatabase, *,
                 swank_cache: dict) -> Optional[Swank]:
        if isinstance(jam, (numbers.Number, str, type(None), bool)):
            return jam
        elif isinstance(jam, list):
            return [self.from_jam(item, swank_database, swank_cache=swank_cache) for item in jam]
        else:
            assert isinstance(jam, dict)
            ((name, savvy_content),) = jam.items()
            assert isinstance(savvy_content, list)
            if name == 'base64.b64decode':
                return base64.b64decode(*savvy_content)
            type_ = utils.name_to_type(name)
            if issubclass(type_, Swank):
                jam_id_name, jam_index = savvy_content
                pair = (JamId(jam_id_name), jam_index)
                try:
                    return swank_cache[pair]
                except KeyError:
                    swank_cache[pair] = swank_cache = swank_database.load_swank(type_, *pair)
                    return swank_cache
            elif issubclass(type_, type):
                result = utils.name_to_type(savvy_content[0])
                assert type(result) is type_
                return result
            else:
                return type_(*(self.from_jam(item, swank_database,
                                             swank_cache=swank_cache) for item in savvy_content))

    def to_jam(self, value: Any, swank_database: SwankDatabase) -> Jam:
        if isinstance(value, (numbers.Number, str, type(None), bool)):
            return value
        elif isinstance(value, list):
            return [self.to_jam(item, swank_database) for item in value]
        else:
            if isinstance(value, Swank):
                if not value.has_jam_id_and_index:
                    value.save()
                assert value.has_jam_id_and_index
            key = 'base64.b64decode' if type(value) is bytes else utils.type_to_name(type(value))
            return {key: [self.to_jam(item, swank_database)
                          for item in object_to_savvy_content(value)]}

