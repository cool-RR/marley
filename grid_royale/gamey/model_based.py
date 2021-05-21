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
                    Sequence, Callable)
import dataclasses

import more_itertools
import keras.models
import tensorflow as tf
import numpy as np

from grid_royale import gamey
from .policing import Policy
from .base import Observation, Action
from . import utils


@dataclasses.dataclass(order=True, frozen=True)
class ActionObservation(utils.NiceDataclass):
    action: Optional[Action]
    observation: Observation



class ModelBasedEpisodicLearningPolicy(Policy):
    '''
    Model-based episodic learning policy.

    This policy assumes we're playing full episodes to the end, and there is no reward
    discounting.
    '''
    def __init__(self, curiosity: numbers.Real = 2) -> None:
        self.reward_map = RewardMap()
        self.curiosity = curiosity
        self.action_observation_chains_lists = collections.defaultdict(list)

    def get_next_action(self, observation: Observation) -> Action:
        return max(
            observation.legal_actions,
            key=lambda action: self.reward_map.get_ucb(
                observation, action, curiosity=self.curiosity
            )
        )

    def train(self, narratives: Sequence[gamey.Narrative]) -> ModelBasedEpisodicLearningPolicy:
        raise NotImplementedError

        action_observation_chains = self.action_observation_chains_lists[observation]
        try:
            action_observation_chain = action_observation_chains.pop()
        except IndexError:
            action_observation_chain = [ActionObservation(None, observation)]

        action_observation_chain.append(ActionObservation(action, next_observation))

        if next_observation.state.is_end:
            total_reward = 0
            for new_action_observation, old_action_observation in \
                                   utils.iterate_windowed_pairs(reversed(action_observation_chain)):
                total_reward += new_action_observation.observation.reward
                self.reward_map.add_sample(old_action_observation.observation,
                                           new_action_observation.action, total_reward)
        else:
            self.action_observation_chains_lists[next_observation].append(action_observation_chain)



def _zero_maker():
    return 0


class RewardMap(collections.abc.Mapping):
    def __init__(self) -> None:
        self._reward_values = collections.defaultdict(_zero_maker)
        self._n_samples = collections.defaultdict(_zero_maker)
        self.n_total_samples = 0

    __len__ = lambda self: len(self._reward_values)
    __iter__ = lambda self: iter(self._reward_values)

    def __getitem__(self, observation_and_action: Iterable) -> numbers.Real:
        return self._reward_values[self._to_key(*observation_and_action)]

    def _to_key(self, observation: Observation, action: Action) -> Tuple[Observation, Action]:
        assert isinstance(observation, Observation)
        assert isinstance(action, Action)
        return (observation, action)



    def add_sample(self, observation: Observation, action: Action, reward: numbers.Real) -> None:
        key = self._to_key(observation, action)
        self._reward_values[key] = (
            self._reward_values[key] *
            (self._n_samples[key] / (self._n_samples[key] + 1)) +
            reward * (1 / (self._n_samples[key] + 1))
        )
        self._n_samples[key] += 1
        self.n_total_samples += 1

    def get_ucb(self, observation: Observation, action: Action,
                curiosity: numbers.Real) -> numbers.Real:
        key = self._to_key(observation, action)
        return self._reward_values[key] + curiosity * math.sqrt(
            utils.cute_div(
                math.log(self.n_total_samples + 2),
                self._n_samples[key]
            )
        )


