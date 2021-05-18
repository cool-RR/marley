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
from .base import PlayerId, Action, Observation, Story
from .policing import Policy


class BaseAggregate(collections.abc.Mapping):
    _aggregate_value_type: Type

    def __init__(self, player_id_to_value: Union[Mapping[PlayerId, Any], Iterable]) -> None:
        self.__player_id_to_value = ImmutableDict(player_id_to_value)
        assert all(isinstance(value, self._aggregate_value_type)
                   for value in self.__player_id_to_value.values())

    def __getitem__(self, player_id: PlayerId) -> Any:
        return self.__player_id_to_value[player_id]

    def __iter__(self) -> Iterator:
        return iter(self.__player_id_to_value)

    def __len__(self) -> int:
        return len(self.__player_id_to_value)

    def __add__(self, other: BaseAggregate):
        if not isinstance(other, BaseAggregate):
            raise NotImplementedError
        to_tuple = lambda x: (x if isinstance(x, tuple) else (x,))
        return _CombinedAggregatePlayerValue(
            (player_id, (*to_tuple(value), *to_tuple(other[player_id]))) for
            player_id, value in self.items()
        )

    @classmethod
    def make_solo(cls, item: Union[numbers.Number, Action, Policy, Observation], /):
        return cls({None: item})

    def get_single(self):
        return more_itertools.one(self.values())

    def __repr__(self):
        return (f'<{type(self).__name__} with {len(self)} '
                f'{self._aggregate_value_type.__name__} objects>')



class _CombinedAggregatePlayerValue(BaseAggregate):
    _aggregate_value_type = tuple

class Activity(BaseAggregate):
    _aggregate_value_type = Action

class Payoff(BaseAggregate):
    _aggregate_value_type = numbers.Number

    def __init__(self, player_id_to_reward: Mapping[PlayerId, numbers.Number]) -> None:
        BaseAggregate.__init__(self, player_id_to_reward)

    @staticmethod
    def make_zero(aggregate_player_value: BaseAggregate) -> Payoff:
        return Payoff(zip(aggregate_player_value, itertools.repeat(0)))



class Culture(BaseAggregate):
    _aggregate_value_type = Policy

    def __init__(self, player_id_to_policy: Mapping[PlayerId, Policy]) -> None:
        BaseAggregate.__init__(self, player_id_to_policy)

    def get_next_activity(self, state: State) -> Activity:
        return Activity({
            player_id: policy.get_next_action(observation) for
            player_id, (policy, observation) in (self + state).items()
        })

    def get_next_culture(self, state: State, activity: Activity, payoff: Payoff,
                         next_state: State) -> Culture:
        return Culture({
            player_id: policy.get_next_policy(Story(old_observation, action,
                                                    reward, new_observation)) for
            player_id, (policy, old_observation, action, reward, new_observation) in
                                             (self + state + activity + payoff + next_state).items()
        })

    def train(self, make_initial_state: Callable[[], State], n_games: int = 20,
              n_states_per_game: int = 30) -> Culture:
        from .gaming import Game
        culture = self
        for _ in range(n_games):
            game = Game.from_state_culture(make_initial_state(), culture)
            game.crunch(n_states_per_game)
            culture = game.cultures[-1]
            yield culture


class State(BaseAggregate):
    _aggregate_value_type = Observation
    is_end: bool

    def __init__(self, player_id_to_observation: Mapping[PlayerId, Observation]) -> None:
        BaseAggregate.__init__(self, player_id_to_observation)


    @staticmethod
    @abc.abstractmethod
    def make_initial(*args, **kwargs) -> State:
        '''Create an initial world state that we can start playing with.'''
        raise NotImplementedError

    @abc.abstractmethod
    def get_next_payoff_and_state(self, activity: Activity) -> Tuple[Payoff, State]:
        raise NotImplementedError

    @staticmethod
    def make_solo(solo_state: SoloState, /):
        assert isinstance(solo_state, SoloState)
        return solo_state


class SoloState(State, Observation):
    def __init__(self):
        BaseAggregate.__init__(self, {None: self})
        self.state = self

    def get_next_payoff_and_state(self, activity: Activity) -> Tuple[Payoff, State]:
        reward, state = self.get_next_reward_and_state(more_itertools.one(activity.values()))
        payoff = Payoff({None: reward})
        return (payoff, state)

    @abc.abstractmethod
    def get_next_reward_and_state(self, action: Action) -> Tuple[numbers.Number, SoloState]:
        raise NotImplementedError


