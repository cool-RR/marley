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
from marley.sharknado.utils import sleep_until
import marley
from marley.sharknado import Interval, Point, IntCrowd




class WhateverJobMixin:

    def __init__(self, base_folder: Union[str, os.PathLike]) -> None:
        self.base_folder: pathlib.Path = pathlib.Path(base_folder).resolve()
        self.serial_folder = self.base_folder / 'serial'
        self.parallel_folder = self.base_folder / 'parallel'
        self.main_path = self.base_folder / 'main'

class WhateverFatJobMixin(WhateverJobMixin):
    dimensions = 1
    def _reduce(self) -> tuple:
        return (type(self), self.base_folder)

class SerialJob(WhateverFatJobMixin, sharknado.SerialJob):

    def get_parent_job_to_weight(self) -> Dict[sharknado.Job, sharknado.Weight]:
        return {ParallelJob(self.base_folder): sharknado.CalfWeight()}

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        assert fat_gain.job == self
        ((i,),) = fat_gain.int_crowd
        if i >= 1:
            previous_serial_path = self.serial_folder / str(i - 1)
            assert previous_serial_path.exists()
            assert previous_serial_path.read_text() == f'serial {str(i - 1)}'

        assert (self.parallel_folder / str(i)).read_text() == f'parallel {str(i)}'

        path = (self.serial_folder / str(i))
        assert not path.exists()
        self.serial_folder.mkdir(parents=True, exist_ok=True)
        path.write_text(f'serial {str(i)}')


class ParallelJob(WhateverFatJobMixin, sharknado.ParallelJob):

    def get_parent_job_to_weight(self) -> Dict[sharknado.Job, sharknado.Weight]:
        return {SerialJob(self.base_folder): sharknado.CalfWeight(-1)}

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        assert fat_gain.job == self
        ((i,),) = fat_gain.int_crowd
        if i >= 1:
            previous_serial_path = self.serial_folder / str(i - 1)
            assert previous_serial_path.exists()
            assert previous_serial_path.read_text() == f'serial {str(i - 1)}'

        path = (self.parallel_folder / str(i))
        assert not path.exists()
        self.parallel_folder.mkdir(parents=True, exist_ok=True)
        path.write_text(f'parallel {str(i)}')


class MainJob(WhateverJobMixin, sharknado.ThinJob):
    def __init__(self, folder: Union[str, os.PathLike], n: int = 10) -> None:
        WhateverJobMixin.__init__(self, folder)
        self.n = n

    def _reduce(self) -> tuple:
        return (type(self), self.base_folder, self.n)

    def get_parent_job_to_weight(self) -> Dict[sharknado.Job, sharknado.Weight]:
        return {
            SerialJob(self.base_folder): sharknado.TadpoleWeight(self.n)
        }

    def thin_run(self) -> None:
        for i in range(self.n):
            assert (self.serial_folder / str(i)).read_text() == f'serial {str(i)}'
            assert (self.parallel_folder / str(i)).read_text() == f'parallel {str(i)}'
        self.base_folder.mkdir(parents=True, exist_ok=True)
        assert not self.main_path.exists()
        self.main_path.write_text('main')



def test_wedge() -> None:
    with marley.utils.create_temp_folder() as temp_folder:
        temp_folder: pathlib.Path
        folder = temp_folder / '0'
        shark = sharknado.Shark(use_multiprocessing=False)
        antilles = shark.supervisor.antilles
        wedge = antilles.wedge

        main_job = MainJob(folder)
        serial_job = SerialJob(folder)
        parallel_job = ParallelJob(folder)
        assert not antilles.job_to_desired_gain

        shark.add_directive_thin_jobs(main_job)
        assert set(wedge.graph.nodes) == {main_job, parallel_job, serial_job}
        assert not wedge.get_child_jobs(main_job)
        assert wedge.get_child_jobs(serial_job) == {main_job, parallel_job}
        assert wedge.get_child_jobs(parallel_job) == {serial_job}
        assert tuple(wedge.iterate_predecessor_jobs(main_job)) == (serial_job, parallel_job,)
        assert tuple(wedge.iterate_predecessor_jobs(main_job, include_self=True)) == \
                                                               (main_job, serial_job, parallel_job,)
        assert tuple(wedge.iterate_predecessor_jobs(serial_job)) == (parallel_job,)
        assert tuple(wedge.iterate_predecessor_jobs(serial_job, include_self=True)) == \
                                                                         (serial_job, parallel_job,)
        assert tuple(wedge.iterate_predecessor_jobs(parallel_job)) == (serial_job,)
        assert tuple(wedge.iterate_predecessor_jobs(parallel_job, include_self=True)) == \
                                                                         (parallel_job, serial_job,)

        assert tuple(wedge.iterate_successor_jobs(main_job)) == ()
        assert tuple(wedge.iterate_successor_jobs(main_job, include_self=True)) == (main_job,)
        assert set(wedge.iterate_successor_jobs(serial_job)) == {main_job, parallel_job}
        assert set(wedge.iterate_successor_jobs(serial_job, include_self=True)) == \
                                                                {serial_job, main_job, parallel_job}
        assert tuple(wedge.iterate_successor_jobs(parallel_job)) == (serial_job, main_job)
        assert tuple(wedge.iterate_successor_jobs(parallel_job, include_self=True)) == \
                                                                 (parallel_job, serial_job,main_job)



def test_desired_gain() -> None:
    with marley.utils.create_temp_folder() as temp_folder:
        temp_folder: pathlib.Path
        folder_0 = temp_folder / '0'
        folder_1 = temp_folder / '1'
        with sharknado.Shark(use_multiprocessing=False, sniff_only=True) as shark:
            antilles = shark.supervisor.antilles

            main_job_0 = MainJob(folder_0)
            serial_job_0 = SerialJob(folder_0)
            parallel_job_0 = ParallelJob(folder_0)
            main_job_1 = MainJob(folder_1)
            assert not antilles.job_to_desired_gain

            shark.add_directive_thin_jobs(main_job_0)
            shark.wait_for_sniffs()
            assert antilles.job_to_desired_gain[main_job_0]
            assert len(antilles.job_to_desired_gain) == 3
            assert set(map(type, antilles.job_to_desired_gain)) == {MainJob, ParallelJob,
                                                                    SerialJob}
            assert all(antilles.job_to_desired_gain.values())

            shark.job_to_directive_gain[main_job_1] = False
            shark.wait_for_sniffs()
            assert len(antilles.job_to_desired_gain) == 3

            shark.job_to_directive_gain[main_job_1] = True
            shark.wait_for_sniffs()
            assert len(antilles.job_to_desired_gain) == 6
            assert antilles.job_to_desired_gain[main_job_1]
            saved_items = tuple(sorted(antilles.job_to_desired_gain.items()))

            job_0_serial = SerialJob(folder_0)
            job_0_parallel = ParallelJob(folder_0)
            shark.job_to_directive_gain[job_0_serial] = job_0_serial.create_gain(
                IntCrowd(Interval[0 : 10])
            )
            shark.wait_for_sniffs()
            new_items = tuple(sorted(antilles.job_to_desired_gain.items()))
            assert new_items == saved_items

            del antilles.job_to_directive_gain[main_job_0]
            del antilles.job_to_directive_gain[main_job_1]
            assert len(antilles.job_to_desired_gain) == 2
            assert not antilles.job_to_desired_gain[main_job_0]
            assert not antilles.job_to_desired_gain[main_job_1]
            assert set(antilles.job_to_desired_gain) == {job_0_serial,
                                                         job_0_parallel}

            del shark.job_to_directive_gain[job_0_serial]
            assert not antilles.job_to_desired_gain


@pytest.mark.parametrize('use_multiprocessing', (False, True))
def test_running(use_multiprocessing: bool) -> None:
    with contextlib.ExitStack() as exit_stack:
        temp_folder: pathlib.Path = exit_stack.enter_context(marley.utils.create_temp_folder())
        shark: sharknado.Shark = exit_stack.enter_context(
                                           sharknado.Shark(use_multiprocessing=use_multiprocessing,
                                                           start=True))
        antilles = shark.antilles
        folder_0: pathlib.Path = temp_folder / '0'
        main_job_0 = MainJob(folder_0, 10)
        serial_job_0 = SerialJob(folder_0)
        parallel_job_0 = ParallelJob(folder_0)
        serial_folder_0 = main_job_0.serial_folder
        parallel_folder_0 = main_job_0.parallel_folder

        folder_1: pathlib.Path = temp_folder / '1'
        main_job_1 = MainJob(folder_1, 20)
        serial_job_1 = SerialJob(folder_1)
        parallel_job_1 = ParallelJob(folder_1)
        serial_folder_1 = main_job_1.serial_folder
        parallel_folder_1 = main_job_1.parallel_folder

        assert not folder_0.exists()
        assert not folder_1.exists()

        shark.add_directive_thin_jobs(main_job_0)

        change_tracker = sharknado.utils.ChangeTracker()
        sharknado.utils.sleep_until(
            lambda: antilles.job_to_finished_gain[main_job_0],
            150,
            reset_condition=lambda: change_tracker(antilles._snapshot())
        )

        assert len(tuple(folder_0.iterdir())) == 3
        assert not folder_1.exists()
        assert (folder_0 / 'main').read_text() == 'main'
        assert len(tuple((serial_folder_0.iterdir()))) == 10
        assert len(tuple((parallel_folder_0.iterdir()))) == 10

        shark.job_to_directive_gain.update({
            main_job_1: True,
            parallel_job_0: IntCrowd(Interval[7 : 15]),
        })

        sleep_until(
            lambda: (antilles.job_to_finished_gain[main_job_1] and
                     (Interval[14] in antilles.job_to_finished_gain[parallel_job_1].int_crowd)),
            150,
        )

        assert antilles.job_to_finished_gain[main_job_0]
        assert len(tuple((serial_folder_0.iterdir()))) == 14
        assert len(tuple((parallel_folder_0.iterdir()))) == 15

        assert antilles.job_to_finished_gain[main_job_1]
        assert len(tuple((serial_folder_1.iterdir()))) == 20
        assert len(tuple((parallel_folder_1.iterdir()))) == 20

        shark.sniff_only = True
        shark.job_to_directive_gain[parallel_job_1] = IntCrowd(Interval[25 : 28])
        shark.wait_for_sniffs()

        assert antilles.job_to_desired_gain[parallel_job_1].int_crowd == IntCrowd(Interval[0 : 28])
        assert antilles.job_to_desired_gain[serial_job_1].int_crowd == IntCrowd(Interval[0 : 27])

        shark.sniff_only = False


        sleep_until(
            lambda: (antilles.job_to_finished_gain[main_job_1] and
                     (Interval[27] in antilles.job_to_finished_gain[parallel_job_1].int_crowd)),
            150,
        )

        assert not antilles.job_to_available_gain
        assert len(tuple((serial_folder_1.iterdir()))) == 27
        assert len(tuple((parallel_folder_1.iterdir()))) == 28
