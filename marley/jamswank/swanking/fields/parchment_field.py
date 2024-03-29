# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional, Any
import itertools
import collections.abc

import more_itertools

from ...jamming import BaseJamDatabase, Jam, JamId, JamItem, JamKind, JamParchment
from ..core import Swank, SwankDatabase
from .base_field import BaseField
from .. import utils
from ..exceptions import EmptyJam


class CantExtendBoundedParchmentRef(Exception):
    pass



class ParchmentRef(collections.abc.MutableSequence):
    def __init__(self, swank_database: SwankDatabase, swank_type: Optional[Type[Swank]] = None,
                 jam_id: Union[JamId, str, None] = None, start_index: int = 0,
                 end_index: Optional[int] = None) -> None:
        assert (swank_type, jam_id).count(None) in (0, 2)
        self.swank_database = swank_database
        self.swank_type = swank_type
        self.jam_id = JamId.parse(jam_id)[0] if jam_id is not None else None
        self.start_index = start_index
        self.end_index = end_index
        self.cached_swanks = {}
        self.cached_modified_indices = set()
        self.cached_length = None

    def migrate(self, jam_id: Union[JamId, str, None] = None, start_index: int = 0,
                end_index: Optional[int] = None, *, delete_old_jam_parchment: bool = True) -> None:
        assert self.is_specified
        old_jam_parchment = self.jam_parchment

        for i, swank in enumerate(self):
            jam_index = i + start_index
            if end_index is not None:
                assert jam_index < end_index
            swank.jam_id = jam_id
            swank.jam_index = jam_index
            swank.save()
            self.cached_modified_indices.discard(i)
        assert not self.cached_modified_indices

        self.jam_id = jam_id
        self.start_index = start_index
        self.end_index = end_index

        self.save()

        if delete_old_jam_parchment and old_jam_parchment != self.jam_parchment:
            old_jam_parchment.delete()


    def save(self) -> None:
        while self.cached_modified_indices:
            i = self.cached_modified_indices.pop()
            self.cached_swanks[i].save()
        self.cached_swanks.clear()

    @property
    def is_specified(self) -> bool:
        count = (self.swank_type, self.jam_id).count(None)
        assert count in (0, 2)
        return not count

    @property
    def is_bounded(self) -> bool:
        result = (self.end_index is not None)
        return result

    def specify(self, swank_or_type: Union[Swank, Type[Swank]]) -> None:
        swank_type = type(swank_or_type) if isinstance(swank_or_type, Swank) else swank_or_type
        self.swank_type = swank_type
        self.jam_id = JamId.create(block_size=swank_type.default_block_size)


    @property
    def jam_kind_name(self) -> Optional[str]:
        if self.is_specified:
            return utils.type_to_name(self.swank_type)
        else:
            return None

    @property
    def jam_parchment(self) -> Optional[JamParchment]:
        if self.is_specified:
            return self.swank_database.jam_database[self.swank_type][self.jam_id]
        else:
            return None

    def __getitem__(self, i: Union[int, slice]) -> Swank:
        if isinstance(i, slice):
            raise NotImplementedError
        if not self.is_specified:
            raise IndexError(i)
        if i < 0:
            i += len(self)
            if not 0 <= i < len(self):
                raise IndexError(i)
        if self.is_bounded and i >= (self.end_index - self.start_index):
            raise IndexError(i)
        try:
            return self.cached_swanks[i]
        except KeyError:
            try:
                self.cached_swanks[i] = self.swank_database.load_swank(
                    self.swank_type, self.jam_id, i + self.start_index
                )
            except EmptyJam as empty_jam:
                raise IndexError(i) from empty_jam
            else:
                return self.cached_swanks[i]

    def __setitem__(self, i: Union[int, slice], swank: Swank) -> None:
        if isinstance(i, slice):
            raise NotImplementedError
        if i < 0:
            raise NotImplementedError
        if not self.is_specified:
            self.specify(type(swank))

        if self.is_bounded and i >= (self.end_index - self.start_index):
            raise IndexError(i)

        swank.jam_id = self.jam_id
        swank.jam_index = i + self.start_index
        self.cached_swanks[i] = swank
        self.cached_modified_indices.add(i)
        if (self.cached_length is None) or i >= self.cached_length:
            self.cached_length = i + 1

    def __len__(self) -> int:
        if self.cached_length is None:
            if self.is_bounded:
                self.cached_length = self.end_index - self.start_index
            else:
                self.cached_length = ((len(self.jam_parchment) - self.start_index)
                                      if self.is_specified else 0)
        return self.cached_length


    def insert(self, i: int, swank: Swank) -> None:
        raise NotImplementedError

    def __delitem__(self, i: int) -> None:
        raise NotImplementedError

    def append(self, swank: Swank) -> None:
        self.extend((swank,))

    def extend(self, swanks: Iterable[Swank]) -> None:
        if self.is_bounded:
            raise CantExtendBoundedParchmentRef
        first_run = True
        for i, swank in enumerate(swanks, start=len(self)):
            if first_run and (not self.is_specified):
                self.swank_type = type(swank)
                self.jam_id = JamId.create(block_size=type(swank).default_block_size)

            first_run = False

            assert isinstance(swank, self.swank_type)
            swank.jam_id = self.jam_id
            swank.jam_index = i + self.start_index

            self.cached_swanks[i] = swank
            self.cached_modified_indices.add(i)
            self.cached_length += 1

    def clear(self):
        raise NotImplementedError

    def reverse(self):
        raise NotImplementedError

    def pop(self, index=-1):
        raise NotImplementedError

    def remove(self, value):
        raise NotImplementedError

    def __repr__(self) -> str:
        items = (
            str(self.swank_database),
            utils.type_to_name(self.swank_type) if (self.swank_type is not None) else 'None',
            repr(str(self.jam_id)),
            *((str(self.start_index),) if self.start_index != 0 else ()),
            *((str(self.end_index),) if self.start_index is not None else ()),
        )
        return f'{type(self).__name__}({", ".join(items)})'

    def has_index(self, i: int) -> bool:
        try:
            self[i]
        except IndexError:
            return False
        else:
            return True




class ParchmentField(BaseField):

    field_type_name = 'parchment'

    def from_jam(self, jam: Jam, swank_database: SwankDatabase, *,
                 swank_cache: dict) -> ParchmentRef:
        if jam is None:
            return self.get_default_value(swank_database)
        else:
            jam_kind_name, jam_id_name, start_index, end_index = jam
            if jam_kind_name is None:
                assert jam_id_name is None
                assert start_index == 0
                assert end_index is None
                return ParchmentRef(swank_database)
            else:
                return ParchmentRef(swank_database, utils.name_to_type(jam_kind_name),
                                    JamId(jam_id_name), start_index, end_index)

    def to_jam(self, value: Optional[ParchmentRef], swank_database: SwankDatabase) -> Jam:
        return (
            value.jam_kind_name,
            (str(value.jam_id) if value.jam_id else None),
            value.start_index,
            value.end_index,
        )

    def get_default_value(self, swank_database: SwankDatabase) -> ParchmentRef:
        return ParchmentRef(swank_database)

    def __set__(self, swank: Swank, other_swanks: Union[Iterable[Swank], ParchmentRef]) -> None:
        if isinstance(other_swanks, ParchmentRef):
            swank._Swank__field_values[self.name] = other_swanks
        else:
            other_swanks = tuple(other_swanks)
            if not other_swanks:
                if self.name not in swank._Swank__field_values:
                    swank._Swank__field_values[self.name] = ParchmentRef(swank.swank_database)
                elif swank._Swank__field_values[self.name]:
                    raise NotImplementedError
                else:
                    pass # Assigned empty to empty, we'll just no-op.
                return
            other_swank_type = type(other_swanks[0])
            assert all((type(other_swank) is other_swank_type) for other_swank in other_swanks)
            new_parchment_ref = ParchmentRef(
                swank.swank_database, other_swank_type,
                JamId.create(block_size=other_swank_type.default_block_size)
            )
            swank._Swank__field_values[self.name] = new_parchment_ref
            new_parchment_ref.extend(other_swanks)

