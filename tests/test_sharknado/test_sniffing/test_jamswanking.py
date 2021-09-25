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
from marley import jamswank
from marley.jamswank.swanking import (Swank, SimpleField, SavvyField,
                                            ParchmentField, SwankDatabase, JamFileDatabase)


class Bar(Swank):
    number = SimpleField()
    number_string = SimpleField()

class Foo(Swank):
    bars = ParchmentField()
    int_crowd = SavvyField(IntCrowd)

class MySwankDatabase(SwankDatabase):
    pass

class CreateBarsJob(sharknado.ParallelJob):

    dimensions = 1

    def __init__(self, foo: Foo) -> None:
        self.foo = foo

    def _reduce(self) -> tuple:
        return (type(self), self.foo)

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        with self.foo.parchment_lock:
            self.foo = self.foo.reload()
            if not self.foo.bars.is_specified:
                self.foo.bars.specify(Bar)
                self.foo.save()

        for point in fat_gain.int_crowd:
            (number,) = point
            self.foo.bars[number] = bar = Bar(swank_database=self.foo.swank_database,
                                              number=number, number_string=str(number))
            bar.save()

        with self.foo.parchment_lock:
            self.foo = self.foo.reload()
            # print(f'Old IntCrowd: {self.foo.int_crowd}')
            self.foo.int_crowd |= fat_gain.int_crowd
            self.foo.save()
            # print(f'New IntCrowd: {self.foo.int_crowd}')


    def fat_sniff(self, fat_gain: sharknado.FatGain) -> Optional[IntCrowd]:
        with self.foo.parchment_lock:
            self.foo = self.foo.reload()
        return fat_gain.int_crowd & self.foo.int_crowd

    def _manual_sniff(self, fat_gain: sharknado.FatGain) -> Optional[IntCrowd]:
        with self.foo.parchment_lock:
            self.foo = self.foo.reload()
            existing_numbers = []
            for point in fat_gain.int_crowd:
                (number,) = point
                try:
                    bar = self.foo.bars[number]
                except IndexError:
                    pass
                else:
                    assert bar.number == number
                    assert bar.number_string == str(number)
                    existing_numbers.append(number)
            return IntCrowd(Interval[number] for number in existing_numbers)



@pytest.mark.parametrize('use_multiprocessing', (False, True))
def test_jamswanking(use_multiprocessing: bool) -> None:

    with contextlib.ExitStack() as exit_stack:
        temp_folder = exit_stack.enter_context(marley.utils.create_temp_folder())
        jam_file_database = JamFileDatabase(temp_folder)
        my_swank_database = MySwankDatabase(jam_file_database)
        foo = Foo(swank_database=my_swank_database)
        foo.save()
        job = CreateBarsJob(foo)
        big_gain = job.create_gain(IntCrowd(Interval[0:100]))
        assert not big_gain.sniff().int_crowd

        with sharknado.Shark(use_multiprocessing=use_multiprocessing, name='0') as shark_0:
            shark_0.job_to_directive_gain[job] = IntCrowd((Interval[4:7], Interval[20:30]))
        with foo.parchment_lock:
            assert (big_gain.sniff().int_crowd == job._manual_sniff(big_gain) ==
                                                         IntCrowd((Interval[4:7], Interval[20:30])))

        with sharknado.Shark(use_multiprocessing=use_multiprocessing, name='1',
                             sniff_only=True) as shark_1:
            shark_1.job_to_directive_gain[job] = IntCrowd(Interval[15:35])
            shark_1.wait_for_sniffs()
            assert shark_1.antilles.job_to_available_gain[job].int_crowd == IntCrowd(
                (Interval[15:20], Interval[30:35])
            )

        # Now let's actually run the job:
        with sharknado.Shark(use_multiprocessing=use_multiprocessing, name='2') as shark_2:
            shark_2.job_to_directive_gain[job] = IntCrowd(Interval[15:35])

        with foo.parchment_lock:
            assert (big_gain.sniff().int_crowd == job._manual_sniff(big_gain) ==
                                                         IntCrowd((Interval[4:7], Interval[15:35])))

