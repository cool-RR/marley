# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import time
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import os
import pathlib
import contextlib

import more_itertools
import pytest

from marley import sharknado
import marley
from marley.sharknado import Interval, IntCrowd


class WriteNumberJob(sharknado.ParallelJob):

    dimensions = 1
    max_int_crowd = IntCrowd(Interval[0 : 256])

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self.path: pathlib.Path = pathlib.Path(path).resolve()

    def _reduce(self) -> tuple:
        return (type(self), self.path)

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        self.path.touch()
        with self.path.open('br+') as file:
            for point in fat_gain.int_crowd:
                (number,) = point
                assert 0 <= number <= 255
                file.seek(number)
                file.write(bytes((number,)))

    def fat_sniff(self, fat_gain: sharknado.FatGain) -> Optional[IntCrowd]:
        assert fat_gain.job == self
        assert fat_gain.int_crowd in self.max_int_crowd
        with contextlib.ExitStack() as exit_stack:
            try:
                file = exit_stack.enter_context(self.path.open('br'))
            except FileNotFoundError:
                return None
            chunk = file.read(256)
            return IntCrowd(
                (Interval[i] for i, character in enumerate(chunk) if i == character)
            )




@pytest.mark.parametrize('use_multiprocessing', (False, True))
def test_fat_gain(use_multiprocessing: bool) -> None:

    with contextlib.ExitStack() as exit_stack:
        temp_folder = exit_stack.enter_context(marley.utils.create_temp_folder())
        paths = tuple(temp_folder / name for name in ('0', '1', '2'))
        (job_0, job_1, job_2) = map(WriteNumberJob, paths)
        job_0: WriteNumberJob
        assert not job_0.create_gain(job_0.max_int_crowd).sniff()
        assert not job_1.create_gain(job_0.max_int_crowd).sniff()
        assert not job_2.create_gain(job_0.max_int_crowd).sniff()
        with sharknado.Shark(use_multiprocessing=use_multiprocessing, name='0') as shark_0:
            shark_0.job_to_directive_gain[job_0] = IntCrowd(Interval[0 : 100])
        assert job_0.create_gain(job_0.max_int_crowd).sniff().int_crowd == \
                                                                         IntCrowd(Interval[0 : 100])

        old_job_0_mtime = job_0.path.stat().st_mtime
        with sharknado.Shark(use_multiprocessing=use_multiprocessing, name='1') as shark_1:
            # We're asking it to do a gain that was already done:
            shark_1.job_to_directive_gain[job_0] = IntCrowd(Interval[0 : 100])
            # The Shark just sniffs and sees there's nothing to do there.
        new_job_0_mtime = job_0.path.stat().st_mtime
        assert old_job_0_mtime == new_job_0_mtime # File wasn't written to.

        with sharknado.Shark(use_multiprocessing=use_multiprocessing, name='2') as shark_2:
            shark_2.job_to_directive_gain[job_1] = IntCrowd((Interval[20:30], Interval[50:70]))

        assert job_1.create_gain(job_1.max_int_crowd).sniff().int_crowd == \
                                           IntCrowd((Interval[0], Interval[20:30], Interval[50:70]))
        # We're getting the 0 item for free, because it's a null byte

        assert not job_2.create_gain(job_0.max_int_crowd).sniff()


