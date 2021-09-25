# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import time
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import os
import pathlib
import logging
import contextlib

import more_itertools
import pytest

from marley import sharknado
import marley
from marley.sharknado import Interval, IntCrowd

logger = logging.getLogger(__name__)


GOLDEN_RATIO = (1 + 5 ** 0.5) / 2



class JobMixin:

    def __init__(self, base_folder: Union[str, os.PathLike]) -> None:
        self.base_folder: pathlib.Path = pathlib.Path(base_folder).resolve()
        self.fibonacci_folder = self.base_folder / 'fibonacci'
        self.golden_ratio_folder = self.base_folder / 'golden_ratio'
        self.main_path = self.base_folder / 'main'

    dimensions = 1
    def _reduce(self) -> tuple:
        return (type(self), self.base_folder)

class FibonacciSerialJob(JobMixin, sharknado.SerialJob):

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        assert fat_gain.job == self
        ((i,),) = fat_gain.int_crowd
        if i in (0, 1):
            value = 1
        else:
            value = (int((self.fibonacci_folder / str(i - 2)).read_text()) +
                     int((self.fibonacci_folder / str(i - 1)).read_text()))

        self.fibonacci_folder.mkdir(exist_ok=True)
        output = str(value)
        output_file = self.fibonacci_folder / str(i)

        output_file.write_text(output)
        logger.debug(f'Wrote {repr(output)} to {output_file}')
        time.sleep(2)
        assert output_file.read_text() == output
        logger.debug(f'Confirmed {output_file} contains {repr(output)}')


class GoldenRatioParallelJob(JobMixin, sharknado.ParallelJob):

    def __init__(self, base_folder: Union[str, os.PathLike]) -> None:
        JobMixin.__init__(self, base_folder)
        self.fibonacci_serial_job = FibonacciSerialJob(base_folder)

    def get_parent_job_to_weight(self) -> Dict[sharknado.Job, sharknado.Weight]:
        return {self.fibonacci_serial_job: sharknado.CalfWeight(1)}

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        assert fat_gain.job == self
        ((i,),) = fat_gain.int_crowd
        value = (int((self.fibonacci_folder / str(i + 1)).read_text()) /
                                                  int((self.fibonacci_folder / str(i)).read_text()))
        self.golden_ratio_folder.mkdir(exist_ok=True)
        (self.golden_ratio_folder / str(i)).write_text((f'{value:.5f}'))



@pytest.mark.parametrize('use_multiprocessing', (False, True))
def test(use_multiprocessing: bool) -> None:
    with contextlib.ExitStack() as exit_stack:
        temp_folder: pathlib.Path = exit_stack.enter_context(marley.utils.create_temp_folder())
        shark: sharknado.Shark = exit_stack.enter_context(
                                           sharknado.Shark(use_multiprocessing=use_multiprocessing,
                                                           start=True, sniff_only=True))
        antilles = shark.antilles
        golden_ratio_parallel_job = GoldenRatioParallelJob(temp_folder)
        fibonacci_serial_job = golden_ratio_parallel_job.fibonacci_serial_job
        n_desired_golden_ratio = 5

        shark.job_to_directive_gain[golden_ratio_parallel_job] = \
                                                sharknado.IntCrowd(Interval[n_desired_golden_ratio])
        shark.wait_for_sniffs()
        assert antilles.job_to_desired_gain[golden_ratio_parallel_job].int_crowd == \
                                                sharknado.IntCrowd(Interval[n_desired_golden_ratio])
        assert antilles.job_to_desired_gain[fibonacci_serial_job].int_crowd == \
                                        sharknado.IntCrowd(Interval[0 : n_desired_golden_ratio + 2])

        shark.sniff_only = False
        path = golden_ratio_parallel_job.golden_ratio_folder / str(n_desired_golden_ratio)

        change_tracker = sharknado.utils.ChangeTracker()
        sharknado.utils.sleep_until(
            lambda: (Interval[n_desired_golden_ratio] in
                          shark.antilles.job_to_finished_gain[golden_ratio_parallel_job].int_crowd),
            total_seconds=100,
            reset_condition=lambda: change_tracker(antilles._snapshot())
        )

        assert abs(float(path.read_text()) - GOLDEN_RATIO) <= 0.01
        assert len(tuple(golden_ratio_parallel_job.golden_ratio_folder.iterdir())) == 1
        assert len(tuple(golden_ratio_parallel_job.fibonacci_folder.iterdir())) == \
                                                                          n_desired_golden_ratio + 2

