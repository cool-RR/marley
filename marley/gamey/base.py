# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import math
import inspect
import re
import abc
import random
import itertools
import collections.abc
import statistics
import concurrent.futures
import enum
import functools
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import dataclasses

import more_itertools
import numpy as np

from marley.utils import ImmutableDict
from marley.jamswank.swanking import SimpleField, SavvyField, SwankField
from marley.gamey.gamey_swanking import GameySwank
from . import utils
from . import exceptions



class _ActionType(abc.ABCMeta):# collections.abc.Sequence):
    __iter__ = lambda cls: iter(cls.all_actions)
    __len__ = lambda cls: len(cls.all_actions)
    def __getitem__(cls, i: int):
        if i >= len(cls):
            raise IndexError
        for j, item in enumerate(cls):
            if j == i:
                return cls
        raise RuntimeError

    @property
    def n_neurons(cls) -> int:
        try:
            return cls._n_neurons
        except AttributeError:
            cls._n_neurons = len(cls)
            return cls._n_neurons



_action_regex_head = re.compile(r'[A-Za-z0-9.]')
_action_regex_tail = re.compile(r'[A-Za-z0-9_.\-/>]*')
_action_regex = re.compile(f'^{_action_regex_head.pattern}'
                           f'{_action_regex_tail.pattern}$')

@functools.total_ordering
class Action(metaclass=_ActionType):
    all_actions: Sequence[Action]
    n_neurons: int

    def __lt__(self, other):
        return self.all_actions.index(self) < self.all_actions.index(other)

    def slugify(self) -> str:
        raw = str(self)
        first_letter = raw[0]
        prefix = '' if _action_regex_head.fullmatch(first_letter) else '0'
        characters = ((c if _action_regex_tail.fullmatch(c) else '-') for c in raw)
        result = f'{prefix}{"".join(characters)}'
        assert _action_regex.fullmatch(result)
        return result

    def to_neural(self) -> np.ndarray:
        # Implementation for simple discrete actions. Can override.
        try:
            return self._to_neural
        except AttributeError:
            self._to_neural = np.array([(self == action) for action in type(self)],
                                        dtype=bool)
            return self._to_neural

    @classmethod
    def from_neural(cls, neural: Iterable) -> Action:
        # Implementation for simple discrete actions. Can override.
        return cls[tuple(neural).index(1)]


class _ActionEnumType(type(Action), type(enum.Enum)):
    pass


class ActionEnum(Action, enum.Enum, metaclass=_ActionEnumType):
    # todo: use in Blackjack
    pass


class Observation(abc.ABC):
    state: State
    legal_actions: Tuple[Action, ...]
    neural_dtype: np.dtype

    @property
    def neural(self) -> np.ndarray:
        '''A structured array of numbers that represents the observation for a neural network.'''
        try:
            return self._neural
        except AttributeError:
            self._neural = self._to_neural()
            return self._neural

    @abc.abstractmethod
    def _to_neural(self) -> np.ndarray:
        '''Represent the observation as a structured array of numbers for a neural network.'''
        pass

    @abc.abstractmethod
    def to_savvy_content(self) -> tuple:
        pass



PlayerId = TypeVar('PlayerId', bound=Hashable)


@dataclasses.dataclass(order=True, frozen=True)
class Story:
    old_observation: Observation
    action: Action
    reward: numbers.Number
    new_observation: Observation


from .aggregating import State