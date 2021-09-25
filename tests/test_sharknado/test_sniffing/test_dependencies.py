# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import time
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import os
import pathlib
import re
import contextlib

import more_itertools
import pytest

from marley import sharknado
from marley.sharknado import IntCrowd, Interval
import marley



class BaseJob(sharknado.ParallelJob):

    dimensions = 1
    prefix: str
    max_int_crowd = IntCrowd(Interval[0:100])

    def __init__(self, folder: Union[str, os.PathLike]) -> None:
        self.folder: pathlib.Path = pathlib.Path(folder).resolve()

    def _reduce(self) -> tuple:
        return (type(self), self.folder)

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        self.folder.mkdir(exist_ok=True)
        for point in fat_gain.int_crowd:
            (number,) = point
            path = self.folder / f'{self.prefix}-{number}'
            path.touch()

    def fat_sniff(self, fat_gain: sharknado.FatGain) -> Optional[IntCrowd]:
        if not self.folder.exists():
            return None
        return IntCrowd(
            Interval[number] for path in self.folder.iterdir() if (
                (match := re.fullmatch(f'^{self.prefix}-([0-9]+)', path.name)) and
                Interval[(number := int(match.group(1)))] in fat_gain.int_crowd
            )
        )


class FooJob(BaseJob):
    prefix = 'foo'
    def get_parent_job_to_weight(self):
        return {BarJob(self.folder): sharknado.CalfWeight()}

class BarJob(BaseJob):
    prefix = 'bar'


@pytest.mark.parametrize('use_multiprocessing', (False, True))
def test_dependencies(use_multiprocessing: bool) -> None:

    with contextlib.ExitStack() as exit_stack:
        temp_folder = exit_stack.enter_context(marley.utils.create_temp_folder())
        folders = tuple(temp_folder / name for name in ('0', '1', '2'))

        (foo_job_0, foo_job_1, foo_job_2) = map(FooJob, folders)
        (bar_job_0, bar_job_1, bar_job_2) = map(BarJob, folders)

        assert not foo_job_0.create_gain(foo_job_0.max_int_crowd).sniff()
        assert not foo_job_1.create_gain(foo_job_0.max_int_crowd).sniff()
        assert not foo_job_2.create_gain(foo_job_0.max_int_crowd).sniff()
        assert not bar_job_0.create_gain(bar_job_0.max_int_crowd).sniff()
        assert not bar_job_1.create_gain(bar_job_0.max_int_crowd).sniff()
        assert not bar_job_2.create_gain(bar_job_0.max_int_crowd).sniff()

        with sharknado.Shark(use_multiprocessing=use_multiprocessing) as shark_0:
            shark_0.job_to_directive_gain[foo_job_0] = IntCrowd(Interval[5:13])
        assert foo_job_0.create_gain(foo_job_0.max_int_crowd).sniff().int_crowd == \
                                                                            IntCrowd(Interval[5:13])
        assert bar_job_0.create_gain(bar_job_0.max_int_crowd).sniff().int_crowd == \
                                                                            IntCrowd(Interval[5:13])

        with sharknado.Shark(use_multiprocessing=use_multiprocessing, sniff_only=True) as shark_1:
            shark_1.wait_for_sniffs()
            assert not shark_1.antilles.job_to_desired_gain
            shark_1.job_to_directive_gain[foo_job_0] = IntCrowd(Interval[5:13])
            shark_1.wait_for_sniffs()
            assert not shark_1.antilles.job_to_desired_gain[bar_job_0]
            shark_1.job_to_directive_gain[foo_job_0] = IntCrowd(Interval[5:14])
            shark_1.wait_for_sniffs()
            assert dict(shark_1.antilles.job_to_desired_gain) == {
                foo_job_0: foo_job_0.create_gain(IntCrowd(Interval[5:14])),
                bar_job_0: bar_job_0.create_gain(IntCrowd(Interval[13])),
            }
