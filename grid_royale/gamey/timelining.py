# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import operator
import concurrent.futures
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, List)
import collections.abc

import keras.models
import more_itertools
import numpy as np

from .base import Observation, Action, Story
from .policing import Policy, QPolicy
from . import utils


class StoryDoesntFitInTimeline(Exception):
    pass



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


class Timeline(collections.abc.Sequence):

    def __init__(self, full_timeline: List[Story], *, length: int) -> None:
        self._full_timeline = full_timeline
        self.stories = ListView(self._full_timeline, length)
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: Union[int, slice]) -> Story:
        return self.stories[i]

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: len={len(self)}>'



    @staticmethod
    def make_initial(story: Story) -> Timeline:
        return Timeline([story], length=1)

    def _is_last_on_full_timeline(self):
        return self[-1] is self._full_timeline[-1]


    def __add__(self, story: Story) -> Timeline:
        assert isinstance(story, Story)
        if self[-1].new_observation != story.old_observation:
            raise StoryDoesntFitInTimeline
        assert self._is_last_on_full_timeline()
        self._full_timeline.append(story)
        return Timeline(self._full_timeline,
                        length=(self.length + 1))


