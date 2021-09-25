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

from marley import gamey
from marley.utils import ImmutableDict
from marley.jamswank.swanking import Swank, SimpleField, SavvyField, SwankDatabase
from . import utils
from . import exceptions
from .base import PlayerId, Action, Observation, Story
from .policing import Policy
from marley.gamey.gamey_swanking import GameySwank

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


class AggregateMixin(collections.abc.Mapping):
    _aggregate_value_type: Type


    @classmethod
    def __get_player_id_to_value_field_name(cls):
        try:
            return cls.__player_id_to_value_field_name
        except AttributeError:
            lowest_swank_base_type = min(
                (base_class for base_class in cls.mro() if issubclass(base_class, Swank) and
                 getattr(base_class, '_Swank__fields', None)),
                key=lambda base_class: len(base_class.mro())
            )
            ((cls.__player_id_to_value_field_name, field),) = \
                                                       lowest_swank_base_type._Swank__fields.items()
            assert isinstance(field, SavvyField)
            return cls.__player_id_to_value_field_name

    @property
    def __player_id_to_value(self) -> dict:
        return getattr(self, self.__get_player_id_to_value_field_name())

    @__player_id_to_value.setter
    def __player_id_to_value(self, d: dict) -> None:
        return setattr(self, self.__get_player_id_to_value_field_name(), d)

    def __getitem__(self, player_id: PlayerId) -> Any:
        return self.__player_id_to_value[player_id]

    def __iter__(self) -> Iterator:
        return iter(self.__player_id_to_value)

    def __len__(self) -> int:
        return len(self.__player_id_to_value)

    def __add__(self, other: AggregateMixin):
        if not isinstance(other, AggregateMixin):
            raise NotImplementedError
        return _CombinedAggregate((self, other))

    @classmethod
    def make_solo(cls, item: Union[numbers.Number, Action, Policy, Observation], /):
        return cls(**{cls.__get_player_id_to_value_field_name(): {None: item}})

    def get_single(self):
        return more_itertools.one(self.values())

    def __repr__(self):
        return (f'<{type(self).__name__} with {len(self)} '
                f'{pluralize(self._aggregate_value_type.__name__)}>')


class _CombinedAggregate(collections.abc.Mapping):
    def __init__(self, aggregates: Iterable[Union[AggregateMixin,
                                                  _CombinedAggregate]]) -> None:
        aggregates = tuple(aggregates)
        assert len(set(map(len, aggregates))) == 1 # They're all the same length.
        self.aggregates = []
        for aggregate in aggregates:
            if isinstance(aggregate, AggregateMixin):
                self.aggregates.append(aggregate)
            else:
                assert isinstance(aggregate, _CombinedAggregate)
                self.aggregates.extend(aggregate.aggregates)


    def __getitem__(self, player_id: PlayerId) -> Any:
        return tuple(aggregate[player_id] for aggregate in self.aggregates)

    def __iter__(self) -> Iterator:
        return iter(self.aggregates[0])

    def __len__(self) -> int:
        return len(self.aggregates[0])


class Activity(AggregateMixin, GameySwank):
    _aggregate_value_type = Action
    player_id_to_action = SavvyField(lambda: {})


class Payoff(AggregateMixin, GameySwank):
    _aggregate_value_type = numbers.Number
    player_id_to_reward = SavvyField(lambda: {})

    @staticmethod
    def make_zero(aggregate_player_value: AggregateMixin) -> Payoff:
        return Payoff(zip(aggregate_player_value, itertools.repeat(0)))



class Culture(AggregateMixin, GameySwank):
    _aggregate_value_type = Policy
    player_id_to_policy = SavvyField(lambda: {})

    def save(self) -> None:
        for policy in self.values():
            policy.save()
        GameySwank.save(self)


    def get_next_activity(self, state: State) -> Activity:
        return Activity(
            player_id_to_action={
                player_id: policy.get_next_action(observation) for
                player_id, (policy, observation) in (self + state).items()
            }
        )

    def train(self, make_initial_state: Callable[[], gamey.State], *, n_games: int = 1_000,
              max_game_length: Optional[int] = None, n_phases: int = 10) -> Policy:
        return more_itertools.last(
            self.train_iterate(
                make_initial_state, n_games=n_games, max_game_length=max_game_length,
                n_phases=n_phases
            )
        )


    def train_iterate(self, make_initial_state: Callable[[], gamey.State], *, n_games: int = 1_000,
                      max_game_length: Optional[int] = None, n_phases: Optional[int] = None,
                      log_average_reward: bool = True) -> Iterable[Policy]:
        from .gaming import Game
        culture = self
        games = []
        games_buffer_max_size = 10 * n_games
        for i_phase in range(n_phases):
            logger.debug(f'Starting training phase {i_phase} out of {n_phases or "infinity"}')
            new_games = tuple(Game.from_state_culture(make_initial_state(), culture)
                              for _ in range(n_games))
            Game.multi_crunch(new_games, n=max_game_length)

            if log_average_reward:
                average_reward = statistics.mean(
                    itertools.chain.from_iterable(new_game.total_payoff.values() for new_game
                                                  in new_games)
                )
                logger.debug(f'The average reward per agent per game is {average_reward:.2f}')

            games.extend(new_games)
            if len(games) >= games_buffer_max_size: # Truncate to avoid memory leaks
                del games[: -games_buffer_max_size]

            culture = type(self)(
                player_id_to_policy={
                    player_id: policy.train(tuple(game.narratives[player_id] for game in games))
                    for player_id, policy in culture.items()
                }
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


class State(AggregateMixin, GameySwank):
    _aggregate_value_type = Observation
    player_id_to_observation = SavvyField(lambda: {})

    is_end: bool

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


class SoloState(State):

    def get_next_payoff_and_state(self, activity: Activity) -> Tuple[Payoff, State]:
        reward, state = self.get_next_reward_and_state(more_itertools.one(activity.values()))
        payoff = Payoff.make_solo(reward)
        return (payoff, state)

    @abc.abstractmethod
    def get_next_reward_and_state(self, action: Action) -> Tuple[numbers.Number, SoloState]:
        raise NotImplementedError
