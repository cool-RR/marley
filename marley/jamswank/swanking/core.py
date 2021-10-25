# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional, Any
import itertools
import functools
import filelock
import re
import collections.abc
import contextlib


from ..jamming import BaseJamDatabase, Jam, JamId, JamItem, JamKind, JamParchment, JamFileDatabase
from . import utils
from .exceptions import EmptyJam
from ..file_locking import FileLock




class SwankDatabase:
    def __init__(self, jam_database: BaseJamDatabase) -> None:
        self.jam_database = jam_database

    @classmethod
    def create_ethereal(cls) -> SwankDatabase:
        return cls(JamFileDatabase.create_ethereal())

    def get_jam_kind(self, arg: Union[Swank, Type[Swank], str]) -> JamKind:
        fixed_arg = type(arg) if isinstance(arg, Swank) else arg
        return self.jam_database[fixed_arg]

    def get_jam_item(self, swank: Swank) -> JamKind:
        assert swank.has_jam_id_and_index
        return self.get_jam_kind(swank)[swank.jam_id][swank.jam_index]

    def get_swank_type(self, arg: Union[Swank, Type[Swank], str]) -> JamKind:
        if isinstance(arg, str):
            jam_kind_name = arg
            return utils.name_to_type(jam_kind_name)
        else:
            if isinstance(arg, Swank):
                return type(arg)
            elif issubclass(arg, Swank):
                return arg


    def load_swank(self, swank_type_or_name: Union[Type[Swank], str], jam_id: Union[JamId, str],
                   jam_index: int) -> Swank:
        jam_kind = self.get_jam_kind(swank_type_or_name)
        swank_type = self.get_swank_type(swank_type_or_name)
        jam_item = jam_kind[jam_id][jam_index]
        jam = jam_item.read_jam()
        if jam is None:
            raise EmptyJam
        return swank_type._Swank__from_jam(jam, jam_id=jam_id, jam_index=jam_index,
                                           swank_database=self)

    def save_swank(self, swank: Swank) -> None:
        if not swank.has_jam_id_and_index:
            swank.jam_id = JamId.create(block_size=swank.default_block_size)
            swank.jam_index = 0
        jam_item = self.get_jam_item(swank)
        jam = swank._Swank__to_jam()
        jam_item.write_jam(jam)

    def iterate_latest(self, swank_type: Type[Swank]) -> Iterable[Swank]:
        return swank_type.iterate_latest(self)

    def _reduce(self) -> tuple:
        return (type(self), self.jam_database)

    def __eq__(self, other: Any) -> bool:
        return (type(other) == type(self)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.jam_database})'




class SwankType(abc.ABCMeta):
    def __new__(mcls, name, bases, dict_) -> Type[Swank]:
        cls = abc.ABCMeta.__new__(mcls, name, bases, dict_)
        if len(cls.mro()) >= 3: # i.e. it's not the base `Swank` type, which doesn't have fields
            cls._Swank__fields = mcls.__get_fields(cls)
        return cls

    def __get_fields(cls) -> dict:
        fields = {}
        existing_names = set()
        for type_ in cls.mro():
            for name, value in vars(type_).items():
                if name in existing_names:
                    # This attribute was already overridden in a subclass, if it's a field we don't
                    # want to include it.
                    continue
                existing_names.add(name)
                if name.startswith('__'):
                    continue
                if isinstance(value, BaseField):
                    assert name not in fields
                    fields[name] = value
        return fields

    def iterate_latest(cls, swank_database: SwankDatabase) -> Iterable[Swank]:
        jam_parchments_by_latest = sorted(
            swank_database.get_jam_kind(cls),
            key=lambda jam_parcment: jam_parcment._get_path().stat().st_mtime,
            reverse=True,
        )
        for jam_parchment in jam_parchments_by_latest:
            for i in reversed(range(len(jam_parchment))):
                yield swank_database.load_swank(cls, jam_parchment.jam_id, i)

    def get_last(cls, swank_database: SwankDatabase) -> Swank:
        try:
            return next(cls.iterate_latest(swank_database))
        except StopIteration:
            raise IndexError

    def get_by_name(cls, swank_database: SwankDatabase, name_substring: str) -> Swank:
        for swank in cls.iterate_latest(swank_database):
            if name_substring in str(swank.jam_id):
                return swank
        raise LookupError



class Swank(metaclass=SwankType):
    default_block_size: int = 1_000
    jam_id: Optional[JamId] = None
    jam_index: Optional[int] = None

    def __init__(self, *, swank_database: SwankDatabase, jam_id: Optional[Union[JamId, str]] = None,
                 jam_index: Optional[int] = None, **kwargs):
        assert (jam_id, jam_index).count(None) in {0, 2}
        self.__field_values = {}
        self.jam_id = None if jam_id is None else JamId.cast(jam_id)
        self.jam_index = jam_index
        self.swank_database = swank_database
        assert set(kwargs) <= set(self._Swank__fields)

        for field_name, field in self.__fields.items():
            try:
                value = kwargs[field_name]
            except KeyError:
                value = field.get_default_value(swank_database)
            setattr(self, field_name, value)


    @property
    def has_jam_id_and_index(self):
        jam_id_exists = (self.jam_id is not None)
        jam_index_exists = (self.jam_index is not None)
        assert jam_id_exists == jam_index_exists
        return jam_id_exists


    @classmethod
    def __from_jam(cls, jam: Jam, *, jam_id: JamId, jam_index: int,
                   swank_database: SwankDatabase) -> Swank:
        fields = cls._Swank__fields
        swank = cls.__new__(cls)
        swank_cache = {(jam_id, jam_index): swank}

        ### Parsing jam into fields: ###############################################################
        #                                                                                          #
        kwargs = {}
        for full_field_name, value in jam.items():
            field_name, field_type_name = full_field_name.split('.')
            field = fields[field_name]
            assert field_type_name == field.field_type_name
            kwargs[field_name] = field.from_jam(value, swank_database=swank_database,
                                                swank_cache=swank_cache)
        #                                                                                          #
        ### Finished parsing jam into fields. ######################################################

        cls.__init__(swank, **kwargs,
                     jam_id=jam_id, jam_index=jam_index, swank_database=swank_database)
        return swank

    def __to_jam(self) -> Jam:
        fields = self._Swank__fields
        return {
            f'{name}.{field_type.field_type_name}':
                field_type.to_jam(getattr(self, name, None), self.swank_database)
            for name, field_type in fields.items()
        }

    @classmethod
    def load(cls, swank_database: SwankDatabase, jam_id: Union[JamId, str],
             jam_index: int) -> Swank:
        return swank_database.load_swank(cls, jam_id=jam_id, jam_index=jam_index)

    def reload(self) -> Swank:
        return type(self).load(self.swank_database, self.jam_id, self.jam_index)

    def save(self, *, all_parchment_fields: bool = True) -> Swank:
        from .fields import ParchmentField
        self.swank_database.save_swank(self)
        if all_parchment_fields:
            for field_name, field in self._Swank__fields.items():
                if isinstance(field, ParchmentField):
                    getattr(self, field_name).save()

    def _reduce(self) -> tuple:
        return (type(self), self.swank_database, self.jam_id, self.jam_index)

    def __eq__(self, other: Any) -> bool:
        return (
            (type(other) == type(self)) and
            self.has_jam_id_and_index and
            (self._reduce() == other._reduce())
        )

    def __hash__(self) -> int:
        return hash(self._reduce())

    def __repr__(self) -> str:
        text = ', '.join(
            f'{name}={value}' for name, value in self.__field_values.items()
        )
        return f'<{type(self).__name__}: {text}>'

    def to_savvy_content(self) -> tuple:
        return (str(self.jam_id), self.jam_index)

    @property
    @functools.cache
    def parchment_lock(self):
        jam_item = self.swank_database.get_jam_item(self)
        lock_path = jam_item.jam_parchment._get_lock_path()
        return FileLock(lock_path)



class SwankRef:
    def __init__(self, swank_database: SwankDatabase, swank_type: Type[Swank], jam_id: JamId,
                 jam_index: int) -> None:
        self.swank_database = swank_database
        self.swank_type = swank_type
        self.swank_type_name = utils.type_to_name(swank_type)
        self.jam_kind_name = utils.type_to_name(swank_type)
        assert isinstance(jam_id, JamId)
        self.jam_id = jam_id
        assert isinstance(jam_index, int)
        self.jam_index = jam_index

    @property
    def jam_item(self) -> JamItem:
        return self.swank_database.jam_database[self.jam_kind_name][self.jam_id][self.jam_index]

    def get(self) -> Swank:
        return self.swank_database.load_swank(self.jam_kind_name, self.jam_id, self.jam_index)

    @staticmethod
    def from_swank_or_ref(swank_or_ref: Optional[Union[Swank, SwankRef]]) -> Optional[SwankRef]:
        if swank_or_ref is None:
            return None
        elif isinstance(swank_or_ref, Swank):
            swank = swank_or_ref
            assert swank.has_jam_id_and_index
            return SwankRef(swank.swank_database, type(swank), swank.jam_id, swank.jam_index)
        else:
            assert isinstance(swank_or_ref, SwankRef)
            return swank_or_ref

    def _reduce(self) -> tuple:
        return (type(self), self.swank_database, self.swank_type, self.jam_id, self.jam_index)

    @property
    @functools.cache
    def parchment_lock(self):
        jam_kind = self.swank_database.get_jam_kind(self.swank_type)
        jam_item = jam_kind[self.jam_id][self.jam_index]
        lock_path = jam_item.jam_parchment._get_lock_path()
        return FileLock(lock_path)

    @contextlib.contextmanager
    def lock_and_load(self, *, save: bool = False) -> Swank:
        with self.parchment_lock:
            yield (swank := self.get())
            if save:
                swank.save()

    @contextlib.contextmanager
    def lock_and_load_or_create(self, *, save: bool = False) -> Swank:
        with self.parchment_lock:
            try:
                swank = self.get()
            except EmptyJam:
                swank = self.swank_type(swank_database=self.swank_database,
                                        jam_id=self.jam_id,
                                        jam_index=self.jam_index)
            yield swank
            if save:
                swank.save()

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.swank_type_name}>'

    def __eq__(self, other: Any) -> bool:
        return (type(other) == type(self)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())








from .fields import BaseField