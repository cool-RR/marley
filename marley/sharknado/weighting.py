# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)


from . import int_crowding
from .int_crowding import IntCrowd, Interval
from .gaining import Gain, ThinGain, FatGain

def clamp_non_negative(number: numbers.Number) -> numbers.Number:
    if number >= 0:
        return number
    else:
        return 0



class Weight(abc.ABC):
    _number: int

    def __init__(self, number: int = 0) -> None:
        self._number = number

    def _reduce(self) -> tuple:
        return (type(self), self._number)

    def __eq__(self, other: Any) -> bool:
        return (type(other) == type(self)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self._number})'


    @abc.abstractmethod
    def get_unblocked_child_gain(self, desired_child_gain: Gain,
                                 finished_parent_gain: Gain) -> Gain:
        raise NotImplementedError

    @abc.abstractmethod
    def get_desired_parent_gain(self, desired_child_gain: Gain, parent_job: Job) -> Gain:
        raise NotImplementedError


class CalfWeight(Weight):
    def __init__(self, offset: int = 0) -> None:
        assert isinstance(offset, int)
        self.offset = offset
        Weight.__init__(self, offset)

    def get_unblocked_child_gain(self, desired_child_gain: Gain,
                                 finished_parent_gain: Gain) -> Gain:
        from .jobbing import ThinJob, FatJob
        is_thin_gain = isinstance(desired_child_gain, ThinGain)
        assert is_thin_gain == isinstance(finished_parent_gain, ThinGain)
        if is_thin_gain:
            return ThinGain(
                desired_child_gain.job,
                (desired_child_gain and finished_parent_gain)
            )
        else:
            assert isinstance(desired_child_gain, FatGain)
            assert isinstance(finished_parent_gain, FatGain)
            main_intervals = tuple(
                Interval(interval.sliceful_tuple[:-1] +
                        (slice(clamp_non_negative(interval.sliceful_tuple[-1].start - self.offset),
                               clamp_non_negative(interval.sliceful_tuple[-1].stop - self.offset)),))
                for interval in finished_parent_gain.int_crowd.intervals
            )
            if self.offset < 0:
                trimmed_int_crowd = desired_child_gain.int_crowd.trim_inclusive()
                bonus_intervals = tuple(
                    Interval(interval.as_tuple() + ((0, -self.offset),))
                    for interval in trimmed_int_crowd.intervals
                )
            else:
                bonus_intervals = ()
            intervals = main_intervals + bonus_intervals
            return desired_child_gain.job.create_gain(desired_child_gain.int_crowd & intervals)

    def get_desired_parent_gain(self, desired_child_gain: Gain, parent_job: Job) -> Gain:
        from .jobbing import ThinJob
        is_thin_gain = isinstance(desired_child_gain, ThinGain)
        assert is_thin_gain == isinstance(parent_job, ThinJob)
        if is_thin_gain:
            return ThinGain(parent_job, desired_child_gain.is_finished)
        else:
            assert isinstance(desired_child_gain, FatGain)
            assert isinstance(parent_job, FatJob)
            return parent_job.create_gain(
                IntCrowd(
                    Interval((
                        interval.sliceful_tuple[:-1] +
                         (slice(clamp_non_negative(interval.sliceful_tuple[-1].start + self.offset),
                               clamp_non_negative(interval.sliceful_tuple[-1].stop + self.offset)),)
                    ))
                     for interval in desired_child_gain.int_crowd.intervals
                )
            )


class TadpoleWeight(Weight):
    def __init__(self, amount: int = 0) -> None:
        assert isinstance(amount, int)
        self.amount = amount
        Weight.__init__(self, amount)

    def get_unblocked_child_gain(self, desired_child_gain: Gain,
                                 finished_parent_gain: FatGain) -> Gain:
        assert isinstance(finished_parent_gain, FatGain)
        if isinstance(desired_child_gain, ThinGain):
            assert finished_parent_gain.job.dimensions == 1
            return ThinGain(desired_child_gain.job,
                            desired_child_gain and (Interval[0 : self.amount] in
                                                    finished_parent_gain.int_crowd))
        else:
            return desired_child_gain.job.create_gain(
                IntCrowd(
                    point for point in desired_child_gain.int_crowd if
                    Interval(point.as_tuple() + ((0, self.amount),))
                    in finished_parent_gain.int_crowd
                )
            )


    def get_desired_parent_gain(self, desired_child_gain: Gain, parent_job: FatJob) -> Gain:
        assert isinstance(parent_job, FatJob)
        if isinstance(desired_child_gain, ThinGain):
            assert parent_job.dimensions == 1
            return parent_job.create_gain(
                IntCrowd(Interval[0 : self.amount]) if desired_child_gain else None
            )
        else:
            return parent_job.create_gain(
                IntCrowd(
                    Interval(interval.as_tuple() + ((0, self.amount),)) for interval in
                    desired_child_gain.int_crowd.intervals
                )
            )

from .jobbing import Job, ThinJob, FatJob
