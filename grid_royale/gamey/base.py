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



@dataclasses.dataclass(order=True, frozen=True)
class ActionObservation(utils.NiceDataclass):
    action: Optional[Action]
    observation: Observation


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
    is_end: bool
    reward: numbers.Real

    @abc.abstractmethod
    def to_neural(self) -> np.ndarray:
        '''Represent the observation as a structured array of numbers for a neural network.'''
        raise NotImplementedError

PlayerId = TypeVar('PlayerId', bound=Hashable)


class State(abc.ABC):
    Observation: Type[Observation]
    Action: Type[Action]
    is_end: bool
    player_id_to_observation: ImmutableDict[PlayerId, Observation]
    player_id_to_strategy: ImmutableDict[PlayerId, strategizing.Strategy]


    def player_id_to_observation_strategy(self):
        try:
            return self._player_id_to_observation_strategy
        except AttributeError:
            self._player_id_to_observation_strategy = ImmutableDict({
                player_id: (observation, self.player_id_to_strategy[player_id]) for
                player_id, observation in self.player_id_to_observation.items()
            })
            return self._player_id_to_observation_strategy


    @abc.abstractmethod
    def get_next_state_from_actions(self, player_id_to_action: Mapping[PlayerId, Action]) -> State:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def make_initial() -> State:
        '''Create an initial world state that we can start playing with.'''
        raise NotImplementedError

    def get_next_state(self) -> State:
        if self.is_end:
            raise exceptions.GameOver
        player_id_to_action = {
            player_id: strategy.decide_action_for_observation(observation)
            for player_id, (observation, strategy) in self.player_id_to_observation_strategy.items()
            if not observation.is_end
        }
        next_state = self.
        get_next_state_from_actions(player_id_to_action)
        for player_id, action in player_id_to_action.items():
            strategy = self.player_id_to_strategy[player_id]
            observation = self.player_id_to_observation[player_id]
            strategy.train(observation, action, next_state.player_id_to_observation[player_id])
        return next_state



class _SinglePlayerStateType(abc.ABCMeta):
    @property
    def Observation(cls) -> _SinglePlayerStateType:
        return cls


class SinglePlayerState(State, Observation, metaclass=_SinglePlayerStateType):

    player_id_to_observation = property(lambda self: ImmutableDict({None: self}))


    @abc.abstractmethod
    def get_next_state_from_action(self, action: Action) -> SinglePlayerState:
        raise NotImplementedError

    def get_next_state_from_actions(self, player_id_to_action: Mapping[PlayerId, Action]) \
                                                                               -> SinglePlayerState:
        return self.get_next_state_from_action(more_itertools.one(player_id_to_action.values()))





from . import strategizing

