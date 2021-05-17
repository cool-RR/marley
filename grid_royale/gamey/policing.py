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
                    Sequence, Callable, Mapping)
import dataclasses
import weakref

import more_itertools
import keras.models
import tensorflow as tf
import numpy as np

from .base import Observation, Action
from . import utils




class Policy(abc.ABC):
    '''
    Logic for deciding which action to take in a given observation.

    Your fancy machine-learning code goes here.
    '''

    is_stubborn: bool

    @abc.abstractmethod
    def get_next_policy(self, story: aggregating.Story) -> Policy:
        # Put your training logic here, if you wish your policy to have training.
        raise NotImplementedError

    @abc.abstractmethod
    def get_next_action(self, observation: Observation) -> Action:
        raise NotImplementedError

    @abc.abstractmethod
    def get_stubborn(self) -> Policy:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self._extra_repr()}'

    def _extra_repr(self) -> str:
        return ('(<...>)' if inspect.signature(self.__init__).parameters else '()')


class CategoricallyStubbornPolicy(Policy):

    is_stubborn =  True

    def get_next_policy(self, story: aggregating.Story) -> CategoricallyStubbornPolicy:
        return self

    def get_stubborn(self) -> CategoricallyStubbornPolicy:
        return self



class SoloPolicy(Policy):
    @property
    @functools.cache
    def culture(self) -> aggregating.Culture:
        return aggregating.Culture.make_solo(self)


class SoloEpisodicPolicy(SoloPolicy):

    def get_score(self, make_initial_state: Callable[[], aggregating.State], n: int = 1_000) -> int:
        stubborn_culture = self.get_stubborn().culture
        games = tuple(gaming.Game.from_state_culture(make_initial_state(), stubborn_culture)
                      for _ in range(n))
        gaming.Game.multi_crunch(games)
        return np.mean(
            tuple(
                sum(payoff.get_single() for payoff in game.payoffs) for game in games
            )
        )


    def train(self, make_initial_state: Callable[[], aggregating.State], n: int = 1_000) -> Policy:
        culture = self.culture
        remaining_n = n
        while remaining_n:
            game = gaming.Game.from_state_culture(make_initial_state(), culture)
            game.crunch(remaining_n)
            remaining_n -= len(game.states)
            culture = game.cultures[-1]
        return culture.get_single()


class RandomPolicy(CategoricallyStubbornPolicy):
    def get_next_action(self, observation: Observation) -> Action:
        return random.choice(observation.legal_actions)


class QPolicy(Policy):
    '''A policy that calculates q-value for observation-actions.'''

    @property
    @functools.cache
    def q_map_cache(self) -> weakref.WeakKeyDictionary:
        return weakref.WeakKeyDictionary()


    @abc.abstractmethod
    def _get_qs_for_observations_uncached(self, observations: Sequence[Observation]) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        raise NotImplementedError

    def get_qs_for_observations(self, observations: Sequence[Observation]) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        try:
            return tuple(self.q_map_cache[observation] for observation in observations)
        except KeyError:
            if any(observation in self.q_map_cache for observation in observations):
                raise NotImplementedError("Can't deal yet with only some of the observations "
                                          "existing in cache.")
        q_maps = self._get_qs_for_observations_uncached(observations)
        for observation, q_map in zip(observations, q_maps):
            self.q_map_cache[observation] = q_map

        return tuple(self.q_map_cache[observation] for observation in observations)

    def get_qs_for_observation(self, observation: Observation) -> Mapping[Action, numbers.Real]:
        return more_itertools.one(self.get_qs_for_observations((observation,)))


from . import aggregating
from . import gaming
