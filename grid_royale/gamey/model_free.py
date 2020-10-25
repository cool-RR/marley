# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import concurrent.futures
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable)
import weakref

import keras.models
import more_itertools
import numpy as np

from .base import Observation, Action, ActionObservation
from .strategizing import Strategy, QStrategy
from . import utils


def _fit_external(model: keras.Model, *args, **kwargs) -> list:
    model.fit(*args, **kwargs)
    return model.get_weights()

class TrainingData:
    def __init__(self, model_free_learning_strategy: ModelFreeLearningStrategy, *,
                  loss: str = 'mse', optimizer: str = 'rmsprop', max_size: int = 10_000) -> None:

        self.model_free_learning_strategy = model_free_learning_strategy
        self.model = keras.models.Sequential(
            layers=(
                keras.layers.Dense(
                    128, activation='relu',
                    input_dim=self.model_free_learning_strategy.State.Observation.n_neurons
                    ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    128, activation='relu',
                    ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    128, activation='relu',
                    ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    self.model_free_learning_strategy.State.Action.n_neurons, # activation='relu'
                    ),
                ),
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        self.max_size = max_size
        self.counter = 0
        self._last_trained_batch = 0
        self.old_observation_neuron_array = np.zeros(
            (max_size, model_free_learning_strategy.State.Observation.n_neurons)
        )
        self.new_observation_neuron_array = np.zeros(
            (max_size, model_free_learning_strategy.State.Observation.n_neurons)
        )
        self.action_neuron_array = np.zeros(
            (max_size, model_free_learning_strategy.State.Action.n_neurons)
        )
        self.reward_array = np.zeros(max_size)
        self.are_not_end_array = np.zeros(max_size)
        self.other_training_data = self

    def add_and_maybe_train(self, observation: Observation, action: Action,
                            next_observation: Observation) -> None:
        self.old_observation_neuron_array[self.counter_modulo] = observation.to_neurons()
        self.action_neuron_array[self.counter_modulo] = action.to_neurons()
        self.new_observation_neuron_array[self.counter_modulo] = next_observation.to_neurons()
        self.reward_array[self.counter_modulo] = next_observation.reward
        self.are_not_end_array[self.counter_modulo] = int(not next_observation.is_end)
        self.counter += 1

        if self.is_training_time():

            n_actions = len(self.model_free_learning_strategy.State.Action)
            slicer = ((lambda x: x) if self.filled_max_size else
                      (lambda x: x[:self.counter_modulo]))
            old_observation_neurons = slicer(self.old_observation_neuron_array)
            new_observation_neurons = slicer(self.new_observation_neuron_array)
            action_neurons = slicer(self.action_neuron_array)
            are_not_ends = slicer(self.are_not_end_array)
            rewards = slicer(self.reward_array)
            n_data_points = old_observation_neurons.shape[0]

            prediction = self.model.predict(
                np.concatenate((old_observation_neurons, new_observation_neurons))
            )
            wip_q_values, new_q_values = np.split(prediction, 2)
            new_other_q_values = self.other_training_data.model.predict(
                new_observation_neurons
            )

            # Assumes discrete actions:
            action_indices = np.dot(action_neurons, range(n_actions)).astype(np.int32)

            batch_index = np.arange(n_data_points, dtype=np.int32)
            wip_q_values[batch_index, action_indices] = (
                rewards + self.model_free_learning_strategy.gamma * are_not_ends *
                new_other_q_values[np.arange(new_q_values.shape[0]),
                                   np.argmax(new_q_values, axis=1)]

            )

            fit_arguments = {
                'x': old_observation_neurons,
                'y': wip_q_values,
                'epochs': max(10, int(self.model_free_learning_strategy.n_epochs *
                                      (n_data_points / self.max_size))),
                'verbose': 0,
            }

            self.model.fit(**fit_arguments)

            self.mark_trained()




    def is_training_time(self) -> bool:
        n_batches = self.counter // self.model_free_learning_strategy.training_batch_size
        return n_batches > self._last_trained_batch


    def mark_trained(self) -> None:
        self._last_trained_batch = \
                               self.counter // self.model_free_learning_strategy.training_batch_size
        assert not self.is_training_time()

    @property
    def counter_modulo(self) -> int:
        return self.counter % self.max_size

    @property
    def filled_max_size(self) -> bool:
        return self.counter >= self.max_size






class ModelFreeLearningStrategy(QStrategy):
    def __init__(self, *, epsilon: numbers.Real = 0.3, gamma: numbers.Real = 0.9,
                 training_batch_size: int = 100, n_epochs: int = 50, n_models: int = 2) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_epochs = n_epochs
        self._fit_future: Optional[concurrent.futures.Future] = None

        self.training_batch_size = training_batch_size
        self.training_datas = tuple(TrainingData(self) for _ in range(n_models))
        for left_training_data, right_training_data in utils.iterate_windowed_pairs(
                                                   self.training_datas + (self.training_datas[0],)):
            left_training_data.other_training_data = right_training_data

        self.q_map_cache = weakref.WeakKeyDictionary()



    def train(self, observation: Observation, action: Action,
              next_observation: Observation) -> None:

        training_data = random.choice(self.training_datas)
        training_data.add_and_maybe_train(observation, action, next_observation)


    def get_qs_for_observations(self, observations: Optional[Sequence[Observation]] = None, *,
                                 observations_neurons: Optional[np.ndarray] = None) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        if observations is None:
            assert observations_neurons is not None
            input_array = observations_neurons
            check_action_legality = False
        else:
            assert observations_neurons is None
            input_array = np.concatenate(
                [observation.to_neurons()[np.newaxis, :] for observation in observations]
            )
            check_action_legality = True
        prediction_output = np.mean(np.array([training_datas.model.predict(input_array) for
                                               training_datas in self.training_datas]), axis=0)
        actions = tuple(self.State.Action)
        if check_action_legality:
            return tuple(
                {action: q for action, q in dict(zip(actions, output_row)).items()
                 if (action in observation.legal_actions)}
                for observation, output_row in zip(observations, prediction_output)
            )
        else:
            return tuple(
                {action: q for action, q in dict(zip(actions, output_row)).items()}
                for output_row in prediction_output
            )



    def decide_action_for_observation(self, observation: Observation, *,
                                       forced_epsilon: Optional[numbers.Real] = None) -> Action:
        epsilon = self.epsilon if forced_epsilon is None else forced_epsilon
        if 0 < epsilon > random.random(): # Clever shortcut-logic optimization.
            return random.choice(observation.legal_actions)
        else:
            try:
                q_map = self.q_map_cache[observation]
            except KeyError:
                q_map = self.q_map_cache[observation] = self.get_qs_for_observation(observation)
            return max(q_map, key=q_map.__getitem__)


    def get_observation_v(self, observation: Observation,
                           epsilon: Optional[numbers.Real] = None) -> numbers.Real:
        if epsilon is None:
            epsilon = self.epsilon
        q_map = self.get_qs_for_observation(observation)
        return np.average(
            (
                max(q_map.values()),
                np.average(tuple(q_map.values()))
            ),
            weights=(1 - epsilon, epsilon)
        )

    def _extra_repr(self) -> str:
        return f'(<...>, n_models={len(self.training_datas)})'


