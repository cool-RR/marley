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


class SingleJob(sharknado.ThinJob):

    def __init__(self, path: Union[str, os.PathLike]) -> None:
        self.path: pathlib.Path = pathlib.Path(path).resolve()

    def _reduce(self) -> tuple:
        return (type(self), self.path)

    def thin_run(self) -> None:
        with self.path.open('a') as file:
            file.write('Hello!')

    def thin_sniff(self) -> bool:
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return False
        else:
            assert text == 'Hello!'
            return True


@pytest.mark.parametrize('use_multiprocessing', (True, )) # todo: Bring back false
def test_single(use_multiprocessing: bool) -> None:
    with marley.utils.create_temp_folder() as temp_folder:
        path: pathlib.Path = temp_folder / 'single.txt'
        single_job = SingleJob(path)
        assert not path.exists()
        with sharknado.Shark(use_multiprocessing=use_multiprocessing, start=True) as shark_0:
            assert not shark_0.antilles.job_to_finished_gain[single_job]
            shark_0.add_directive_thin_jobs(single_job)
            sleep_until(lambda: shark_0.antilles.job_to_finished_gain[single_job], 150)
            assert path.read_text() == 'Hello!'

        with sharknado.Shark(use_multiprocessing=use_multiprocessing, start=True) as shark_1:
            assert not shark_1.antilles.job_to_finished_gain[single_job]
            shark_1.add_directive_thin_jobs(single_job)
            sleep_until(lambda: shark_0.antilles.job_to_finished_gain[single_job], 150)
            assert path.read_text() == 'Hello!'
