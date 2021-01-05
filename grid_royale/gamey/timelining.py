# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import operator
import concurrent.futures
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable)
import collections.abc

import keras.models
import more_itertools
import numpy as np

from .base import Observation, Action, Story
from .policing import Policy, QPolicy
from . import utils


class StoryDoesntFitInTimeline(Exception):
    pass



class BaseTimeline(collections.abc.Sequence):
    observations: Sequence[Observation]
    actions: Sequence[Action]
    rewards: Sequence[numbers.Number]

    @property
    def sequences(self):
        return (self.observations, self.actions, self.rewards)

    def __len__(self):
        return min(map(len, self.sequences))

    def __getitem__(self, i: Union[int, slice]) -> Any:
        if isinstance(i, int):
            if - (len(self) + 1) < i < len(self):
                fixed_i = i if i >= 0 else (i + len(self))
                assert 0 <= fixed_i < len(self)
                return tuple(sequence[fixed_i] for sequence in self.sequences)
            else:
                raise IndexError
        else:
            assert isinstance(i, slice)
            raise NotImplementedError


class FullTimeline(BaseTimeline):
    def __init__(self, observations: Iterable[Observation], actions: Iterable[Action],
                 rewards: Iterable[numbers.Number]) -> None:
        self.observations = list(observations)
        self.actions = list(actions)
        self.rewards = list(rewards)

class ListView(collections.abc.Sequence):
    def __init__(self, _list: list, length: int) -> None:
        self._list = _list
        self.length = length


    def __len__(self):
        return max(self.length, len(self._list))


    def __getitem__(self, i: Union[int, slice]) -> Any:
        if isinstance(i, int):
            if - (self.length + 1) < i < self.length:
                return self._list[i]
            else:
                raise IndexError
        else:
            assert isinstance(i, slice)
            raise NotImplementedError


class Timeline(BaseTimeline):
    def __init__(self, full_timeline: FullTimeline, *, observations_length: int) -> None:
        self._full_timeline = full_timeline
        self.observations = ListView(self._full_timeline.observations, observations_length)
        self.actions = ListView(self._full_timeline.actions, observations_length - 1)
        self.rewards = ListView(self._full_timeline.rewards, observations_length - 1)


    @staticmethod
    def make_initial(story: Story) -> Timeline:
        return Timeline(
            FullTimeline((story.observation, story.next_observation), (story.action,),
                         (story.reward,)),
            observations_length=2
        )

    def _is_last_on_full_timeline(self):
        return all(map(operator.is_, zip(self[-1], self._full_timeline[-1])))


    def __add__(self, story: Story) -> Timeline:
        assert isinstance(story, Story)
        assert self._is_last_on_full_timeline()
        if story.observation != self.observations[-1]:
            raise StoryDoesntFitInTimeline
        self._full_timeline.observations.append(story.next_observation)
        self._full_timeline.actions.append(story.action)
        self._full_timeline.rewards.append(story.reward)
        assert not self._is_last_on_full_timeline()
        return Timeline(self._full_timeline,
                        observations_length=(self.observations.length + 1))


