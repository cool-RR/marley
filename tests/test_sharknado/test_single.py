# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import time
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import os
import pathlib

import more_itertools
import pytest

from marley import sharknado
import marley


class WriteHelloJob(sharknado.ThinJob):
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self.path: pathlib.Path = pathlib.Path(path).resolve()

    def _reduce(self) -> tuple:
        return (type(self), self.path)

    def thin_run(self) -> None:
        self.path.write_text('Hello')


@pytest.mark.parametrize('use_multiprocessing', (False, True))
def test_single(use_multiprocessing: bool) -> None:

    with marley.utils.create_temp_folder() as temp_folder:
        paths = tuple(temp_folder / name for name in ('yaba', 'daba', 'doo', 'ooo'))
        jobs = tuple(map(WriteHelloJob, paths))

        with sharknado.Shark(use_multiprocessing=use_multiprocessing) as shark:
            shark.add_directive_thin_jobs(jobs[0])
        assert paths[0].read_text() == 'Hello'
        assert not paths[1].exists()
        assert not paths[2].exists()
        assert not paths[3].exists()
        with sharknado.Shark(jobs[1:]) as shark:
            pass
        assert {path.read_text() for path in paths} == {'Hello'}
