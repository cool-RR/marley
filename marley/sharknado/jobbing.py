# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import functools
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)

from .gaining import Gain, ThinGain, FatGain, SerialGain, ParallelGain, GainArg
from .int_crowding import IntCrowd


@functools.total_ordering
class JobType(abc.ABCMeta):
    def __get_long_name(self) -> str:
        return f'{self.__module__}.{self.__name__}'


    def __lt__(self, other: Union[JobType, Any]) -> bool:
        if not isinstance(other, JobType):
            raise NotImplementedError
        return self.__get_long_name() < other.__get_long_name()


@functools.total_ordering
class Job(metaclass=JobType):
    dimensions: int
    gain_type: Type[Gain]

    def get_parent_job_to_weight(self) -> Dict[Job, Weight]:
        return {}

    @abc.abstractmethod
    def _reduce(self) -> Tuple[Any]:
        pass

    def __repr__(self) -> str:
        return f'{type(self).__name__}{repr(self._reduce()[1:])}'

    def __hash__(self) -> int:
        return hash(self._reduce())

    def __eq__(self, other: Union[Any, Job]) -> bool:
        return (type(self) is type(other)) and self._reduce() == other._reduce()

    def create_gain(self, arg: GainArg = None) -> Gain:
        return self.gain_type(self, arg)

    def __lt__(self, other: Union[Job, Any]) -> bool:
        if not isinstance(other, Job):
            raise NotImplementedError
        return self._reduce() < other._reduce()




class ThinJob(Job):
    dimensions = 0
    gain_type = ThinGain

    def thin_sniff(self) -> Union[bool, ThinGain, None]:
        return None

    def thin_run(self) -> Optional[ThinGain]:
        return None

class FatJob(Job):
    max_chunk_size = 1
    dimensions: int

    def fat_sniff(self, fat_gain: FatGain) -> Union[FatGain, IntCrowd, None]:
        return None

    def fat_run(self, fat_gain: FatGain) -> Optional[FatGain]:
        return None


class SerialJob(FatJob):
    gain_type = SerialGain


class ParallelJob(FatJob):
    gain_type = ParallelGain


class JobSniffingJob(ThinJob):
    def __init__(self, target_gain: Gain):
        self.target_gain = target_gain

    def thin_run(self) -> Optional[Gain]:
        return self.target_gain.sniff()

    def thin_sniff(self) -> None:
        raise RuntimeError("Oh you didn't!")

    def _reduce(self) -> tuple:
        return (type(self), self.target_gain)




from .weighting import Weight, CalfWeight, TadpoleWeight
