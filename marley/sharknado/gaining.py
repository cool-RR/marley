# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import itertools
import dataclasses
import logging

import more_itertools
import operator as operator_module

from . import int_crowding
from .int_crowding import IntCrowd, Interval, Point

logger = logging.getLogger(__name__)


GainArg = TypeVar('GainArg', bound=Union[IntCrowd, bool, None])

class CantWorkOnEmptyGain(Exception):
    pass



class Gain(abc.ABC):

    def __init__(self, job: Job, arg: GainArg) -> None:
        self.job = job
        self.arg = arg

    @abc.abstractmethod
    def _reduce(self) -> tuple:
        raise NotImplementedError

    @abc.abstractmethod
    def get_doable(self) -> Gain:
        raise NotImplementedError

    @abc.abstractmethod
    def __bool__(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self) -> Optional[Gain]:
        raise NotImplementedError

    @abc.abstractmethod
    def get_gain_desired_by_self(self) -> Gain:
        '''


        We're assuming that `self` is the desired gain, and we want the gain we desire because of
        that. In other words, this exists for `SerialGain` to demand everything from zero to the
        maximum.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_gain_unblocked_by_finished_gain_of_same_job(self, finished_gain: Gain) -> Gain:
        '''


        We're assuming that `self` is the desired gain, and we're looking for unblocked gain that
        intersects with this desired gain.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def get_gain_unblocked_by_finished_parent_gain(self, finished_parent_gain: Gain,
                                                  parent_weight: Weight) -> Gain:
        '''


        We're assuming that `self` is the desired gain, and we're looking for unblocked gain that
        intersects with this desired gain.
        '''
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        return (type(other) == type(self)) and (self._reduce() == other._reduce())

    def __hash__(self) -> int:
        return hash(self._reduce())

    def __repr__(self) -> str:
        return f'{type(self).__name__}{repr(self._reduce()[1:])}'

    @abc.abstractmethod
    def __contains__(self, other: Gain) -> bool:
        raise NotImplementedError


    @abc.abstractmethod
    def __and__(self, other: Union[Gain, Any]) -> Gain:
        raise NotImplementedError

    @abc.abstractmethod
    def __or__(self, other: Union[Gain, Any]) -> Gain:
        raise NotImplementedError

    @abc.abstractmethod
    def __and__(self, other: Union[Gain, Any]) -> Gain:
        raise NotImplementedError

    @abc.abstractmethod
    def __sub__(self, other: Union[Gain, Any]) -> Gain:
        raise NotImplementedError




class ThinGain(Gain):
    def __init__(self, job: ThinJob, is_finished: Optional[bool] = False) -> None:
        self.is_finished = bool(is_finished)
        Gain.__init__(self, job, self.is_finished)

    def _reduce(self) -> tuple:
        return (type(self), self.job, self.is_finished)

    def get_doable(self) -> ThinGain:
        return self

    def __bool__(self) -> bool:
        return self.is_finished

    def get_gain_desired_by_self(self) -> ThinGain:
        return self

    def get_gain_unblocked_by_finished_gain_of_same_job(self, finished_gain: ThinGain) -> ThinGain:
        return self

    def get_gain_unblocked_by_finished_parent_gain(self, finished_parent_gain: Gain,
                                                  parent_weight: Weight) -> Gain:
        if isinstance(finished_parent_gain, ThinGain):
            assert isinstance(parent_weight, CalfWeight)
            assert parent_weight.offset == 0
            return self
        else:
            assert isinstance(finished_parent_gain, FatGain)
            assert isinstance(parent_weight, TadpoleWeight)
            return ThinGain(
                self.job,
                Interval[0 : parent_weight.amount] in finished_parent_gain.int_crowd
            )

    def __call__(self) -> Optional[ThinGain]:
        logger.debug(f'Executing {self}')
        if self:
            return_value = self.job.thin_run()
            logger.debug(f'Finished executing {self}')
            return return_value


    def sniff(self) -> ThinGain:
        if not self:
            return self
        result = self.job.thin_sniff()
        if result is None or isinstance(result, bool):
            return self.job.create_gain(bool(result))
        else:
            assert isinstance(result, ThinGain)
            assert result.job == self.job
            return result

    def __contains__(self, other: ThinGain) -> bool:
        assert isinstance(other, ThinGain) and (self.job == other.job)
        return bool(self or not other)


    def __or__(self, other: Union[Gain, Any]) -> Gain:
        if (type(self) is type(other)) and (self.job == other.job):
            return type(self)(self.job, self.is_finished or other.is_finished)
        else:
            return NotImplemented

    def __and__(self, other: Union[Gain, Any]) -> Gain:
        if (type(self) is type(other)) and (self.job == other.job):
            return type(self)(self.job, self.is_finished and other.is_finished)
        else:
            return NotImplemented

    def __sub__(self, other: Union[Gain, Any]) -> Gain:
        if (type(self) is type(other)) and (self.job == other.job):
            return type(self)(self.job, self.is_finished and not other.is_finished)
        else:
            return NotImplemented


class FatGain(Gain):
    def __init__(self, job: FatJob, int_crowd: Optional[IntCrowd] = None) -> None:
        self.int_crowd: IntCrowd = int_crowd if (int_crowd is not None) else IntCrowd()
        Gain.__init__(self, job, self.int_crowd)


    def _reduce(self) -> tuple:
        return (type(self), self.job, self.int_crowd)

    def get_doable(self) -> FatGain:
        if not self:
            raise CantWorkOnEmptyGain
        first_interval = more_itertools.first(self.int_crowd.intervals)
        first_point = more_itertools.first(first_interval)
        interesting_interval = Interval(first_point[:-1] + first_interval.as_tuple()[-1:])
        money_item = interesting_interval.as_tuple()[-1]
        if isinstance(money_item, int):
            result_interval = interesting_interval
        else:
            start = money_item.start
            end = money_item.stop
            best_end = min(end, start + self.job.max_chunk_size)
            result_interval = Interval(interesting_interval.as_tuple()[:-1] + ((start, best_end),))

        return self.job.create_gain(IntCrowd(result_interval))

    def get_gain_unblocked_by_finished_parent_gain(self, finished_parent_gain: FatGain,
                                                   parent_weight: Weight) -> Gain:
        return parent_weight.get_unblocked_child_gain(self, finished_parent_gain)


    def __call__(self) -> Optional[FatGain]:
        logger.debug(f'Executing {self}')
        if self:
            return_value = self.job.fat_run(self)
            logger.debug(f'Finished executing {self}')
            return return_value

    def sniff(self) -> FatGain:
        result = self.job.fat_sniff(self)
        if result is None:
            return self.job.create_gain()
        elif isinstance(result, IntCrowd):
            return self.job.create_gain(result)
        else:
            assert isinstance(result, FatGain)
            assert result.job == self.job
            return result




    def __contains__(self, other: FatGain) -> bool:
        assert isinstance(other, FatGain) and (self.job == other.job)
        return other.int_crowd in self.int_crowd

    def __bool__(self) -> bool:
        return bool(self.int_crowd)


    def _arithmetic(self, other: Union[IntCrowd, Any], operator: Callable) -> Gain:
        if (type(self) is type(other)) and (self.job == other.job):
            return type(self)(self.job, operator(self.arg, other.arg))
        else:
            return NotImplemented

    def __or__(self, other: Union[Gain, Any]) -> Gain:
        return self._arithmetic(other, operator_module.or_)

    def __and__(self, other: Union[Gain, Any]) -> Gain:
        return self._arithmetic(other, operator_module.and_)

    def __sub__(self, other: Union[Gain, Any]) -> Gain:
        return self._arithmetic(other, operator_module.sub)


class SerialGain(FatGain):
    def get_gain_desired_by_self(self) -> ThinGain:
        if not self:
            return self
        return self.job.create_gain(
            IntCrowd(
                Interval(interval.as_tuple()[:-1] + ((0, interval.sliceful_tuple[-1].stop),))
                for interval in self.int_crowd.intervals
            )
        )

    def get_gain_unblocked_by_finished_gain_of_same_job(self, finished_gain: Gain) -> Gain:
        return SerialGain(
            self.job,
            IntCrowd(
                (point for point in self.int_crowd if (point[-1] == 0) or
                 Point(point[:-1] + ((point[-1] - 1),)) in finished_gain.int_crowd)
            )
        )



class ParallelGain(FatGain):
    def get_gain_desired_by_self(self) -> ParallelGain:
        return self

    def get_gain_unblocked_by_finished_gain_of_same_job(self, finished_gain: Gain) -> Gain:
        return self


@dataclasses.dataclass(order=True, frozen=True)
class GainDyad:
    requested_gain: Gain
    returned_gain: Optional[Gain]




from .weighting import Weight, CalfWeight, TadpoleWeight
from .jobbing import Job, ThinJob, FatJob
