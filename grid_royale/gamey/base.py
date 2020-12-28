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

from .utils import ImmutableDict
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

    @abc.abstractmethod
    def to_neural(self) -> np.ndarray:
        '''Represent the observation as a structured array of numbers for a neural network.'''
        raise NotImplementedError

PlayerId = TypeVar('PlayerId', bound=Hashable)



class BaseAggregatePlayerValue(collections.abc.Mapping):
    __value_type: Type

    def __init__(self, player_id_to_value: Union[Mapping[PlayerId, Any], Iterable]):
        self.__player_id_to_value = ImmutableDict(player_id_to_value)
        assert all(type(value) == self.__value_type for value in self.__player_id_to_value.values())

    def __getitem__(self, player_id: PlayerId) -> Any:
        return self.__player_id_to_value[player_id]

    def __iter__(self) -> Iterator:
        return iter(self.__player_id_to_value)

    def __len__(self) -> int:
        return len(self.__player_id_to_value)

    def __add__(self, other: BaseAggregatePlayerValue):
        if not isinstance(other, BaseAggregatePlayerValue):
            raise NotImplementedError
        to_tuple = lambda x: (x if isinstance(x, tuple) else (x,))
        return _CombinedAggregatePlayerValue(
            (player_id, (*to_tuple(value), *to_tuple(other[player_id]))) for
            player_id, value in self.items()
        )

class _CombinedAggregatePlayerValue(collections.abc.Mapping):
    __value_type : tuple

class Activity(BaseAggregatePlayerValue):
    __value_type = Action

class Payoff(BaseAggregatePlayerValue):
    __value_type = numbers.Number

    @staticmethod
    def make_zero(aggregate_player_value: BaseAggregatePlayerValue) -> Payoff:
        return Payoff(zip(aggregate_player_value, itertools.repeat(0)))



class Culture(BaseAggregatePlayerValue):
    __value_type = policing.Policy

    def get_next_activity_and_culture(self, game: Game, payoff: Payoff,
                                      state: State) -> Tuple[Activity, Culture]:
        activity_dict = {}
        culture_dict = {}
        for player_id, (policy, reward, observation) in (self + payoff + state):
            policy: policing.Policy
            (activity_dict[player_id], culture_dict[player_id]) = \
                                        policy.get_next_action_and_policy(game, reward, observation)

        return (Activity(activity_dict), Culture(culture_dict))

class State(BaseAggregatePlayerValue):
    __value_type = Observation
    Observation: Type[Observation]
    is_end: bool

    @staticmethod
    @abc.abstractmethod
    def make_initial(*args, **kwargs) -> State:
        '''Create an initial world state that we can start playing with.'''
        raise NotImplementedError

    @abc.abstractmethod
    def get_next_payoff_and_state(self, activity: Activity) -> Tuple[Payoff, State]:
        raise NotImplementedError


class _SoloStateType(abc.ABCMeta):
    @property
    def Observation(cls) -> _SoloStateType:
        return cls

class SoloState(State, Observation, metaclass=_SoloStateType):
    def __init__(self):
        self._BaseAggregatePlayerValue__player_id_to_value = ImmutableDict({None: self})

    def get_next_payoff_and_state(self, activity: Activity) -> Tuple[Payoff, State]:
        reward, state = self.get_next_reward_and_state(more_itertools.one(activity.values()))
        payoff = Payoff({None: reward})
        return (payoff, state)

    @abc.abstractmethod
    def get_next_reward_and_state(self, action: Action) -> Tuple[numbers.Number, SoloState]:
        raise NotImplementedError



from . import policing
from .gaming import Game
