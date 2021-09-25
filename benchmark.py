#!python
#  todo: delete this file
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



def benchmark(use_multiprocessing: bool = True) -> None:
    with contextlib.ExitStack() as exit_stack:
        temp_folder: pathlib.Path = exit_stack.enter_context(marley.utils.create_temp_folder())
        shark: sharknado.Shark = exit_stack.enter_context(
                                           sharknado.Shark(use_multiprocessing=use_multiprocessing,
                                                           start=True))
        folder_0: pathlib.Path = temp_folder / '0'
        main_job_0 = MainJob(folder_0, 500)

        assert not folder_0.exists()

        shark.add_directive_thin_jobs(main_job_0)

        start_time = time.monotonic()
        for _ in range(30):
            if main_job_0.main_path.exists():
                break
            else:
                show_stack_traces()
                time.sleep(0.5)
                show_stack_traces()
        else:
            raise TimeoutError
        end_time = time.monotonic()
        duration = end_time - start_time
        print(f'Duration: {duration:.2f}s')


def show_stack_traces():
    import sys, traceback
    print('=========== Starting printing frames', file=sys.stderr)
    for tid, frame in sys._current_frames().items():
        print('Stack for thread {}'.format(tid), file=sys.stderr)
        traceback.print_stack(frame)
        print('', file=sys.stderr)



if __name__ == '__main__':
    benchmark()
