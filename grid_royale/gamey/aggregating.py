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
import click
import logging

from grid_royale import gamey
from .utils import ImmutableDict
from . import utils
from . import exceptions
from .base import PlayerId, Action, Observation, Story
from .policing import Policy

logger = logging.getLogger(__name__)


def pluralize(s: str) -> str:
    known_values = {
        'policy': 'policies',
        'observation': 'observations',
        'reward': 'rewards',
        'mood': 'moods',
    }
    try:
        return known_values[s.lower()]
    except KeyError:
        return f'{s} objects'


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
                f'{pluralize(self._aggregate_value_type.__name__)}>')



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

    def train(self, make_initial_state: Callable[[], gamey.State], *, n_games: int = 1_000,
              max_game_length: Optional[int] = None, n_phases: int = 10) -> Policy:
        return more_itertools.last(
            self.train_iterate(
                make_initial_state, n_games=n_games, max_game_length=max_game_length,
                n_phases=n_phases
            )
        )


    def train_iterate(self, make_initial_state: Callable[[], gamey.State], *, n_games: int = 1_000,
                      max_game_length: Optional[int] = None,
                      n_phases: Optional[int] = None) -> Iterable[Policy]:
        from .gaming import Game
        culture = self
        games = []
        games_buffer_max_size = 10 * n_games
        for _i_phase in more_itertools.islice_extended(itertools.count())[:n_phases]:
            games.extend(Game.from_state_culture(make_initial_state(), culture)
                         for _ in range(n_games))
            if len(games) >= games_buffer_max_size: # Truncate to avoid memory leaks
                del games[: -games_buffer_max_size]
            Game.multi_crunch(games, n=max_game_length)
            culture = type(self)(
                {player_id: policy.train(tuple(game.narratives[player_id] for game in games))
                 for player_id, policy in culture.items()}
            )
            yield culture

    def train_iterate_progress_bar(self, make_initial_state: Callable[[], gamey.State], *,
                                   label: Optional[str] = None, n_games: int = 1_000,
                                   max_game_length: Optional[int] = None,
                                   n_phases: Optional[int] = None, **kwargs) -> Iterable[Policy]:
        if label is None:
            label = repr(self)
        logger.info(f'Training {label}...')
        progress_bar = click.progressbar(
            self.train_iterate(make_initial_state, n_games=n_games,
                               max_game_length=max_game_length, n_phases=n_phases),
            length=n_phases,
            **kwargs
        )
        with progress_bar:
            yield from progress_bar
        logger.info(f'Finished training {label}.')


    def train_progress_bar(self, make_initial_state: Callable[[], gamey.State], *,
                           label: Optional[str] = None, n_games: int = 1_000,
                           max_game_length: Optional[int] = None, n_phases: Optional[int] = None,
                           **kwargs) -> Iterable[Policy]:
        return more_itertools.last(
            self.train_iterate_progress_bar(
                make_initial_state=make_initial_state, label=label, n_games=n_games,
                max_game_length=max_game_length, n_phases=n_phases, **kwargs
            )
        )


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


