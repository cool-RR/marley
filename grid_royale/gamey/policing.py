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

from .base import Observation, Action
from . import utils




class Policy(abc.ABC):
    '''
    Logic for deciding which action to take in a given observation.

    Your fancy machine-learning code goes here.
    '''

    State: Type[State]

    @abc.abstractmethod
    def get_next_action_and_policy(self, game: Game, reward: numbers.Number,
                                   observation: Observation) -> Tuple(Action, Policy):
        # Put your training logic here, if you wish your policy to have training.
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{type(self).__name__}{self._extra_repr()}'

    def _extra_repr(self) -> str:
        return ('(<...>)' if inspect.signature(self.__init__).parameters else '()')



class SoloPolicy(Policy):
    @property
    @functools.cache
    def culture(self) -> Culture:
        return Culture.make_solo(self)


class SoloEpisodicPolicy(SoloPolicy):

    def get_score(self, make_initial_state: Callable[[], State], n: int = 1_000) -> int:
        scores = []
        for _ in range(n):
            game = Game(self.culture, make_initial_state())
            game.crunch()
            scores.append(sum(payoff.get_single() for payoff in game.payoffs))
        return np.mean(scores)



class RandomPolicy(Policy):
    def get_next_action_and_policy(self, game: Game, reward: numbers.Number,
                                   observation: Observation) -> Tuple(Action, Policy):
        return (random.choice(observation.legal_actions), self)


class QPolicy(Policy):
    '''A policy that calculates q-value for observation-actions.'''

    @abc.abstractmethod
    def get_qs_for_observations(self, observations: Sequence[Observation]) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        raise NotImplementedError

    def get_qs_for_observation(self, observation: Observation) -> Mapping[Action, numbers.Real]:
        return more_itertools.one(self.get_qs_for_observations((observation,)))



from .gaming import Game
from .base import State, Culture