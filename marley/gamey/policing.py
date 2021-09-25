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
import tensorflow as tf
import numpy as np

from marley import gamey
from .base import Observation, Action
from . import utils
from marley.jamswank.swanking import SimpleField, SavvyField, SwankRefField
from marley.gamey.gamey_swanking import GameySwank




class Policy(abc.ABC, GameySwank):
    '''
    Logic for deciding which action to take in a given observation.

    Your fancy machine-learning code goes here.
    '''

    def train(self, narratives: Sequence[gamey.Narrative]) -> Policy:
        # Override this method with your training code
        return self

    @abc.abstractmethod
    def get_next_action(self, observation: Observation) -> Action:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self._extra_repr()}'

    def _extra_repr(self) -> str:
        return ('(<...>)' if inspect.signature(self.__init__).parameters else '()')


class SoloPolicy(Policy):
    @functools.cache
    def make_culture(self) -> aggregating.Culture:
        return aggregating.Culture.make_solo(self)


class SoloEpisodicPolicy(SoloPolicy):

    def get_score(self, make_initial_state: Callable[[], aggregating.State], n: int = 1_000) -> int:
        culture = self.make_culture()
        games = tuple(gaming.Game.from_state_culture(make_initial_state(), culture)
                      for _ in range(n))
        gaming.Game.multi_crunch(games)
        return np.mean(
            tuple(
                sum(payoff.get_single() for payoff in game.payoffs) for game in games
            )
        )


class RandomPolicy(Policy):
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
        result = [self.q_map_cache.get(observation, None) for observation in observations]
        uncached_indices = tuple(i for i, q_map in enumerate(result) if q_map is None)
        uncached_observations = tuple(observations[i] for i in uncached_indices)

        new_q_maps = self._get_qs_for_observations_uncached(uncached_observations)
        for i, observation, new_q_map in zip(uncached_indices, uncached_observations, new_q_maps):
            self.q_map_cache[observation] = new_q_map
            result[i] = new_q_map

        assert None not in result
        return tuple(result)

    def get_qs_for_observation(self, observation: Observation) -> Mapping[Action, numbers.Real]:
        return more_itertools.one(self.get_qs_for_observations((observation,)))


from . import aggregating
from . import gaming
