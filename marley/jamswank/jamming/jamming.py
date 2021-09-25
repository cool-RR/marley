# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import re
import os
import sys
import io
import abc
import itertools
import bisect
import pathlib
import json
import collections.abc
from typing import Union, Iterable, Any, Mapping, TypeVar, Optional, Tuple, Sequence, Sized
import datetime as datetime_module
import string
import random

from . import utils


Jam = TypeVar('Jam', bound=dict)

jam_id_pattern = re.compile(
    r'^([0-9]{8})-([0-9]{6})-([0-9]{6})-([0-9a-z]{30})\.([0-9]{12})$'
)
entry_pack_name_pattern = re.compile('^[0-9]{40}$')
max_entry_pack_size = 1_000
datetime_format = '%Y%m%d-%H%M%S-%f'
digits_and_lowercase_ascii = string.digits + string.ascii_lowercase

class JamException(Exception):
    pass

class JamCorruption(JamException):
    pass

class JamTooLong(JamException):
    pass


class DateTimeSupportingJsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime_module.datetime):
            return o.isoformat()

        return json.JSONEncoder.default(self, o)


def fixed_size_json(jam: Jam, size: int) -> bytes:
    x: bytes = json.dumps(jam, cls=DateTimeSupportingJsonEncoder).encode('utf-8')
    if len(x) > size:
        raise JamTooLong
    else:
        return x.ljust(size, b' ')

def cute_json_load(text: str) -> Any:
    if text.startswith('\0'):
        return None
    else:
        return json.loads(text)



class JamId:
    def __init__(self, jam_id_or_name: Union[JamId, str]) -> None:
        self.text = jam_id_or_name if isinstance(jam_id_or_name, str) else str(jam_id_or_name)
        assert '/' not in self.text
        assert '\\' not in self.text
        match = jam_id_pattern.fullmatch(self.text)
        assert match is not None
        self.block_size = int(match.groups()[-1])

    @property
    def datetime(self):
        try:
            return self._datetime
        except AttributeError:
            self._datetime = datetime_module.datetime.strptime(self.text, datetime_format)
            return self._datetime

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.text)}))'

    @staticmethod
    def parse(jam_id_or_str: Union[str, JamId]) -> Tuple[JamId, str]:
        if isinstance(jam_id_or_str, str):
            jam_id_name = jam_id_or_str
            jam_id = JamId(jam_id_name)
        else:
            assert isinstance(jam_id_or_str, JamId)
            jam_id_name = str(jam_id_or_str)
            jam_id = jam_id_or_str
        return (jam_id, jam_id_name)

    @staticmethod
    def cast(jam_id_or_str: Union[str, JamId]) -> JamId:
        if isinstance(jam_id_or_str, str):
            return JamId(jam_id_or_str)
        else:
            assert isinstance(jam_id_or_str, JamId)
            return jam_id_or_str

    @staticmethod
    def create(*, block_size: int = 1_000) -> JamId:
        assert isinstance(block_size, int)
        assert block_size >= 1
        return JamId(
            f'{datetime_module.datetime.now().strftime(datetime_format)}-'
            f'{"".join(random.choices(digits_and_lowercase_ascii, k=30))}.{block_size:012d}'
        )

    def _reduce(self) -> tuple:
        return (type(self), self.text)


    def __eq__(self, other: JamId) -> bool:
        return (type(self) is type(other)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())





class BaseJamReducable:
    @abc.abstractmethod
    def _reduce(self) -> Tuple[Any]:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        return (type(self) is type(other)) and (self._reduce() == other._reduce())

    def __repr__(self) -> str:
        return f'{type(self).__name__}({", ".join(map(repr, self._reduce()))})'


class BaseJamDatabase(abc.ABC, Iterable, BaseJamReducable):
    def __init__(self, path: Union[pathlib.Path, str]) -> None:
        self.path = pathlib.Path(path)

    def __getitem__(self, name: str) -> JamKind:
        return JamKind(self, name)

    def __iter__(self) -> Iterable[JamKind]:
        yield from self._get_jam_kinds()

    def __len__(self):
        return len(self._get_jam_kinds())

    @abc.abstractmethod
    def _get_jam_kinds(self) -> Sequence[JamKind]:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_jam_kind_length(self, jam_kind_name: str) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_jam_parchments(self, jam_kind_name: str) -> Sequence[JamParchment]:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_jam_parchment_length(self, jam_kind_name: str, jam_id: Union[str, JamId]) -> int:
        raise NotImplementedError


    @abc.abstractmethod
    def _read_texts(self, jam_kind_name: str, jam_id: Union[str, JamId],
                    start: int, end: Optional[int]) -> Tuple[str]:
        raise NotImplementedError


class BaseWritableJamDatabase(BaseJamDatabase):
    @abc.abstractmethod
    def _write_blobs(self, jam_kind_name: str, jam_id: Union[str, JamId],
                     texts: Iterable[bytes], *, start: int = 0, append: bool = False) -> None:
        raise NotImplementedError


class JamKind(BaseJamReducable):
    def __init__(self, jam_database: BaseJamDatabase, name: str) -> None:
        assert re.fullmatch('[a-zA-Z0-9_.]+', name)
        self.jam_database = jam_database
        self.name = name

    def _reduce(self) -> Tuple[Any]:
        return (self.jam_database, self.name)

    def __len__(self) -> int:
        return self.jam_database._get_jam_kind_length(self.name)

    def __iter__(self) -> Iterable[JamParchment]:
        yield from self.jam_database._get_jam_parchments(self.name)

    def __getitem__(self, jam_id: Union[JamId, str]) -> JamParchment:
        return JamParchment(self, JamId(jam_id))


class JamParchment(collections.abc.Sequence, BaseJamReducable):
    def __init__(self, jam_kind: JamKind, jam_id: JamId) -> None:
        self.jam_kind = jam_kind
        self.jam_id = jam_id
        self.block_size = jam_id.block_size
        self.jam_database = self.jam_kind.jam_database

    def _reduce(self) -> Tuple[Any]:
        return (self.jam_database, self.jam_kind, self.jam_id)

    def __iter__(self) -> Iterable[JamItem]:
        for i in itertools.count():
            yield JamItem(self, i)

    def __len__(self) -> int:
        return self.jam_database._get_jam_parchment_length(self.jam_kind.name, self.jam_id)

    def __getitem__(self, i: int) -> JamItem:
        assert isinstance(i, int) # Slices not supported
        return JamItem(self, i)

    def extend_blobs(self, blobs: Iterable[bytes]) -> None:
        assert isinstance(self.jam_database, BaseWritableJamDatabase)
        self.jam_database._write_blobs(self.jam_kind.name, self.jam_id, blobs, append=True)

    def extend_jams(self, jams: Iterable[Jam]) -> None:
        self.extend_blobs(fixed_size_json(jam, self.block_size) for jam in jams)

    def read_texts(self, start: int, end: Optional[int]) -> Tuple[str, ...]:
        return self.jam_database._read_texts(self.jam_kind.name, self.jam_id, start, end)

    def _get_path(self) -> os.PathLike:
        # Todo: This method makes me want to refactor jamming to have the path on each object.
        from .jam_file_database import JamFileDatabase
        assert isinstance(self.jam_database, JamFileDatabase)
        return self.jam_database.path / self.jam_kind.name / str(self.jam_id)

    def _get_lock_path(self) -> os.PathLike:
        path = self._get_path()
        return type(path)(str(path) + '.lock')
        




class JamItem(BaseJamReducable):
    def __init__(self, jam_parchment: JamParchment, i: str) -> None:
        self.jam_parchment = jam_parchment
        self.jam_database = self.jam_parchment.jam_database
        self.i = i

    def _reduce(self) -> Tuple[Any]:
        return (self.jam_database, self.jam_parchment, self.i)

    def read_text(self) -> bytes:
        end = (self.i + 1) if self.i != -1 else None
        texts = self.jam_parchment.read_texts(self.i, end)

        if texts:
            (text,) = texts
            return text
        else:
            raise IndexError(self.i)

    def write_blob(self, blob: bytes) -> None:
        assert isinstance(self.jam_database, BaseWritableJamDatabase)
        self.jam_database._write_blobs(self.jam_parchment.jam_kind.name,
                                       self.jam_parchment.jam_id, (blob,), start=self.i)

    def read_jam(self) -> Jam:
        return cute_json_load(self.read_text())

    def write_jam(self, jam: Jam) -> None:
        self.write_blob(fixed_size_json(jam, self.jam_parchment.block_size))
        
