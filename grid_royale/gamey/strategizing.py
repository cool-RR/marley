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

import more_itertools
import keras.models
import tensorflow as tf
import numpy as np

from .base import Observation, Action, ActionObservation
from . import utils




class Strategy(abc.ABC):
    '''
    Logic for deciding which action to take in a given observation.

    Your fancy machine-learning code goes here.
    '''

    State: Type[State]

    @abc.abstractmethod
    def decide_action_for_observation(self, observation: Observation) -> Action:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self._extra_repr()}'

    def _extra_repr(self) -> str:
        return ('(<...>)' if inspect.signature(self.__init__).parameters else '()')

    def train(self, observation: Observation, action: Action,
              next_observation: Observation) -> None:
        pass # Put your training logic here, if you wish your strategy to have training.


class SinglePlayerStrategy(Strategy):

    def get_score(self, n: int = 1_000, state_factory: Optional[Callable] = None,
                  max_length: Optional[int] = None) -> int:
        from .culturing import SinglePlayerCulture

        single_player_culture = SinglePlayerCulture(self.State, strategy=self)
        return sum(
            single_player_state.reward for single_player_state in single_player_culture.
                         iterate_many_games(n=n, max_length=max_length, state_factory=state_factory)
        )



class RandomStrategy(Strategy):
    def decide_action_for_observation(self, observation: Observation) -> Action:
        return random.choice(observation.legal_actions)


class QStrategy(Strategy):
    '''A strategy that calculates q-value for observation-actions.'''

    # @abc.abstractmethod
    # def get_observation_v(self, observation: Observation) -> numbers.Real:
        # raise NotImplementedError

    @abc.abstractmethod
    def get_qs_for_observations(self, observations: Sequence[Observation]) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        raise NotImplementedError

    def get_qs_for_observation(self, observation: Observation) -> Mapping[Action, numbers.Real]:
        return more_itertools.one(self.get_qs_for_observations((observation,)))




