# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import itertools
import time
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import os
import pathlib

import more_itertools
import pytest

from marley import sharknado
from marley.sharknado.utils import sleep_until
import marley


class WriteHelloParallelJob(sharknado.ParallelJob):

    dimensions = 1

    def __init__(self, folder: Union[str, os.PathLike]) -> None:
        self.folder: pathlib.Path = pathlib.Path(folder).resolve()

    def _reduce(self) -> tuple:
        return (type(self), self.folder)

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        assert fat_gain.job == self
        ((i,),) = fat_gain.int_crowd
        path = (self.folder / str(i))

        if i == 5:
            # Because it's a parallel job, we'll ensure the other files get written without waiting
            # for us to finish:
            expected_paths = tuple(
                (self.folder / str(k)) for k in itertools.chain(range(5), range(6, 10))
            )
            sleep_until(lambda: all(expected_path.exists() for expected_path in expected_paths),
                        1_000,
                        exception=AssertionError('Other paths should have been written by now.'))

        path.write_text('Hello')




class Job(sharknado.ThinJob):
    def __init__(self, folder: Union[str, os.PathLike], n: int = 10) -> None:
        self.folder: pathlib.Path = pathlib.Path(folder).resolve()
        self.n = n

    def _reduce(self) -> tuple:
        return (type(self), self.folder, self.n)

    def get_parent_job_to_weight(self) -> Dict[sharknado.Job, sharknado.Weight]:
        return {
            WriteHelloParallelJob(self.folder): sharknado.TadpoleWeight(self.n)
        }

@pytest.mark.parametrize('use_multiprocessing', (False, True))
def test_parallel(use_multiprocessing: bool) -> None:
    with marley.utils.create_temp_folder() as temp_folder:
        temp_folder: pathlib.Path
        with sharknado.Shark(use_multiprocessing=use_multiprocessing) as shark:
            shark.add_directive_thin_jobs(Job(temp_folder))
        paths = tuple(temp_folder.iterdir())
        assert len(paths) == 10
        assert {path.read_text() for path in paths} == {'Hello'}
        assert {int(path.name) for path in paths} == set(range(10))
