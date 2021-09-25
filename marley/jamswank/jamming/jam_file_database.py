# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import re
import io
import abc
import itertools
import bisect
import pathlib
import collections.abc
from typing import Union, Iterable, Any, Mapping, TypeVar, Optional, Tuple, Sequence, Sized
import datetime as datetime_module
import string
import os
import random

import pyfakefs.fake_pathlib

from . import utils
from .jamming import (BaseWritableJamDatabase, Jam, JamKind, JamId, JamParchment, JamCorruption,
                      jam_id_pattern)


class JamFileDatabase(BaseWritableJamDatabase):
    def __init__(self, path: os.PathLike) -> None:
        self.path = pathlib.Path(path) if isinstance(path, str) else path

    @staticmethod
    def create_ethereal() -> JamFileDatabase:
        fake_filesystem = pyfakefs.fake_filesystem.FakeFilesystem()
        fake_pathlib_module = pyfakefs.fake_pathlib.FakePathlibModule(fake_filesystem)
        return JamFileDatabase(fake_pathlib_module.Path('/'))


    def _reduce(self) -> Tuple[Any]:
        return (self.path,)

    def _get_jam_kind_length(self, jam_kind_name: str) -> int:
        try:
            return utils.iterable_length((self.path / jam_kind_name).iterdir())
        except FileNotFoundError:
            return 0

    def _get_jam_kinds(self) -> Sequence[JamKind]:
        return tuple(self[jam_kind_path.name] for jam_kind_path in self.path.iterdir())

    def _get_jam_parchments(self, jam_kind_name: str) -> Sequence[JamParchment]:
        jam_kind = self[jam_kind_name]
        try:
            return tuple(
                jam_kind[jam_parchment_path.name] for jam_parchment_path in
                (self.path / jam_kind_name).iterdir()
                if jam_id_pattern.fullmatch(jam_parchment_path.name)
            )
        except FileNotFoundError:
            return ()

    def _get_jam_parchment_length(self, jam_kind_name: str, jam_id: Union[str, JamId]) -> int:
        (jam_id, jam_id_name) = JamId.parse(jam_id)
        path = self.path / jam_kind_name / jam_id_name
        try:
            return (path.stat().st_size) // jam_id.block_size
        except FileNotFoundError:
            return 0

    def _open_file_seeked(self, jam_kind_name: str, jam_id: JamId, *, start: int = 0,
                          at_end: bool = False, write: bool = False) -> io.IOBase:
        path = self.path / jam_kind_name / str(jam_id)
        if write:
            path.parent.mkdir(parents=True, exist_ok=True)
        if at_end:
            assert start == 0
            assert write
            file = path.open(f'ba')
            assert file.tell() % jam_id.block_size == 0
        else:
            try:
                file = path.open(f'b{"r+" if write else "r"}')
            except FileNotFoundError:
                path.touch()
                file = path.open(f'b{"r+" if write else "r"}')
            file.seek(jam_id.block_size * start)
        return file

    def _read_texts(self, jam_kind_name: str, jam_id: Union[str, JamId],
                    start: int, end: Optional[int]) -> Tuple[str]:
        (jam_id, _) = JamId.parse(jam_id)
        block_size = jam_id.block_size
        jam_parchment_length = self._get_jam_parchment_length(jam_kind_name, jam_id)
        if start < 0:
            start += jam_parchment_length
        if end is not None and end < 0:
            end += jam_parchment_length
        try:
            file = self._open_file_seeked(jam_kind_name, jam_id, start=start)
        except FileNotFoundError:
            return ()

        with file:
            result = []
            iterator = range(start, end) if (end is not None) else itertools.count(start)
            for _ in iterator:
                data = file.read(block_size)
                if len(data) == block_size:
                    result.append(data.decode('utf-8'))
                elif not data:
                    break
                else:
                    assert 0 < len(data) != block_size
                    raise JamCorruption
            return tuple(result)


    def _write_blobs(self, jam_kind_name: str, jam_id: Union[str, JamId],
                     blobs: Iterable[bytes], *, start: int = 0, append: bool = False) -> None:
        if append:
            assert start == 0
        (jam_id, _) = JamId.parse(jam_id)
        with self._open_file_seeked(jam_kind_name, jam_id, start=start, at_end=append,
                                    write=True) as file:
            for blob in blobs:
                assert len(blob) == jam_id.block_size
                file.write(blob)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(str(self.path))})'

    def __eq__(self, other: Any) -> bool:
        return (type(other) == type(self)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())
