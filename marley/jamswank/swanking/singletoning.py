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
from .core import Swank, SwankType, SwankDatabase


class SingletonSwankType(SwankType):
    @property
    def _singleton_jam_id(cls) -> JamId:
        return JamId.create(block_size=cls.default_block_size, zero=True)



class SingletonSwank(Swank, metaclass=SingletonSwankType):
    def __init__(self, *, swank_database: SwankDatabase, **kwargs):
        self._singleton_jam_id = type(self)._singleton_jam_id

        # Being forgiving yet watchful if wer're given redundant arguments:
        assert kwargs.pop('jam_id', self._singleton_jam_id) == self._singleton_jam_id
        assert kwargs.pop('jam_index', 0) == 0

        Swank.__init__(
            self,
            swank_database=swank_database,
            jam_id=self._singleton_jam_id,
            jam_index=0,
            **kwargs,
        )

    @classmethod
    def load(cls, swank_database: SwankDatabase) -> SingletonSwank:
        return swank_database.load_swank(cls, jam_id=cls._singleton_jam_id, jam_index=0)

    @classmethod
    def load_or_create(cls, swank_database: SwankDatabase) -> SingletonSwank:
        try:
            return cls.load(swank_database)
        except EmptyJam:
            return cls(swank_database=swank_database)

    @classmethod
    @contextlib.contextmanager
    def lock_and_load_or_create(cls, swank_database: SwankDatabase, *,
                                save: bool = False) -> SingletonSwank:
        jam_kind = swank_database.get_jam_kind(cls)
        lock_path = jam_kind[cls._singleton_jam_id]._get_lock_path()
        parchment_lock = FileLock(lock_path)

        with parchment_lock:
            try:
                singleton_swank = cls.load(swank_database)
            except IndexError:
                singleton_swank = cls(swank_database=swank_database)
            yield singleton_swank
            if save:
                singleton_swank.save()

    def reload(self) -> SingletonSwank:
        return type(self).load(self.swank_database)

