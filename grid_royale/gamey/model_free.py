# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import concurrent.futures
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable)

import keras.models
import numpy as np

from .base import Observation, Action, ActionObservation
from .strategizing import Strategy, QStrategy
from . import utils


def _fit_external(model: keras.Model, *args, **kwargs) -> list:
    model.fit(*args, **kwargs)
    return model.get_weights()

class TrainingData:
    def __init__(self, model_free_learning_strategy: ModelFreeLearningStrategy, *,
                 max_size: int = 10_000) -> None:
        self.model_free_learning_strategy = model_free_learning_strategy
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

    def add(self, observation: Observation, action: Action, next_observation: Observation) -> None:
        self.old_observation_neuron_array[self.counter_modulo] = observation.to_neurons()
        self.action_neuron_array[self.counter_modulo] = action.to_neurons()
        self.new_observation_neuron_array[self.counter_modulo] = next_observation.to_neurons()
        self.reward_array[self.counter_modulo] = next_observation.reward
        self.are_not_end_array[self.counter_modulo] = int(not next_observation.is_end)
        self.counter += 1


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
                 training_batch_size: int = 100, loss: str = 'mse', optimizer: str = 'rmsprop',
                 n_epochs: int = 50) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.n_epochs = n_epochs
        self._fit_future: Optional[concurrent.futures.Future] = None

        self.model = keras.models.Sequential(
            layers=(
                keras.layers.Dense(
                    128, activation='relu',
                    input_dim=self.State.Observation.n_neurons
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
                     self.State.Action.n_neurons, # activation='relu'
                ),

            ),
            name='awesome_model'
        )
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        self.training_batch_size = training_batch_size
        self.training_data = TrainingData(self)



    def train(self, observation: Observation, action: Action,
              next_observation: Observation) -> None:

        self.training_data.add(observation, action, next_observation)

        if self.training_data.is_training_time():

            n_actions = len(self.State.Action)
            slicer = ((lambda x: x) if self.training_data.filled_max_size else
                      (lambda x: x[:self.training_data.counter_modulo]))
            old_observation_neurons = slicer(self.training_data.old_observation_neuron_array)
            new_observation_neurons = slicer(self.training_data.new_observation_neuron_array)
            action_neurons = slicer(self.training_data.action_neuron_array)
            are_not_ends = slicer(self.training_data.are_not_end_array)
            rewards = slicer(self.training_data.reward_array)
            n_data_points = old_observation_neurons.shape[0]

            if self._fit_future is not None:
                weights = self._fit_future.result()
                self.model.set_weights(weights)
                self._fit_future = None

            wip_q_values, new_q_values = np.split(
                self.model.predict(
                    np.concatenate((old_observation_neurons, new_observation_neurons))
                ),
                2
            )

            # Assumes discrete actions:
            action_indices = np.dot(action_neurons, range(n_actions)).astype(np.int32)

            batch_index = np.arange(n_data_points, dtype=np.int32)
            wip_q_values[batch_index, action_indices] = (
                rewards + self.gamma * np.max(new_q_values, axis=1) * are_not_ends
            )

            fit_arguments = {
                'x': old_observation_neurons,
                'y': wip_q_values,
                'epochs': max(1, int(self.n_epochs *
                                     (n_data_points / self.training_data.max_size))),
                'verbose': 0,
            }

            # This seems not to work fast:
            # if executor is not None:
                # self._fit_future = executor.submit(_fit_external, self.model, **fit_arguments)
            # else:
            self.model.fit(**fit_arguments)

            self.training_data.mark_trained()


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
        prediction_output = self.model.predict(input_array)
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
                                       forced_epsilon: Optional[numbers.Real] = None,
                                       extra: Optional[np.ndarray] = None) -> Action:
        epsilon = self.epsilon if forced_epsilon is None else forced_epsilon
        if 0 < epsilon > random.random(): # Clever shortcut-logic optimization.
            return random.choice(observation.legal_actions)
        else:
            q_map = self.get_qs_for_observation(observation) if extra is None else extra
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



