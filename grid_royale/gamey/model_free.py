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

from .base import Observation, Action
from .policing import Policy, QPolicy
from . import utils

BATCH_SIZE = 64

def _fit_external(model: keras.Model, *args, **kwargs) -> list:
    model.fit(*args, **kwargs)
    return model.get_weights()

class TrainingData:
    def __init__(self, model_free_learning_policy: ModelFreeLearningPolicy, *,
                 max_size: int = 5_000) -> None:

        self.model_free_learning_policy = model_free_learning_policy
        self.model: Optional[keras.Model] = None

        self.max_size = max_size
        self.counter = 0
        self._last_trained_batch = 0
        self.old_observation_neuron_array = None
        self.new_observation_neuron_array = None
        self.action_neuron_array = None
        self.reward_array = np.zeros(max_size)
        self.are_not_end_array = np.zeros(max_size)
        self.other_training_data = self
        self._created_arrays_and_model = False

    def _create_arrays_and_model(self, observation: Observation, action: Action,):
        self.model = self.model_free_learning_policy.create_model(observation, action)
        observation_neural = observation.to_neural()
        self.old_observation_neuron_array = np.zeros(
            (self.max_size,), dtype=observation_neural.dtype
        )
        self.new_observation_neuron_array = np.zeros(
            self.old_observation_neuron_array.shape,
            dtype=observation_neural.dtype
        )
        self.action_neuron_array = np.zeros(
            (self.max_size, type(action).n_neurons)
        )
        self._created_arrays_and_model = True



    def add_and_maybe_train(self, observation: Observation, action: Action,
                            next_observation: Observation) -> None:
        if not self._created_arrays_and_model:
            self._create_arrays_and_model(observation, action)
        self.old_observation_neuron_array[self.counter_modulo] = observation.to_neural()
        self.action_neuron_array[self.counter_modulo] = action.to_neural()
        self.new_observation_neuron_array[self.counter_modulo] = next_observation.to_neural()
        self.reward_array[self.counter_modulo] = next_observation.reward
        self.are_not_end_array[self.counter_modulo] = int(not next_observation.state.is_end)
        self.counter += 1

        if self.is_training_time():

            n_actions = len(self.model_free_learning_policy.State.Action)

            pre_slicer = ((lambda x: x) if self.filled_max_size else
                          (lambda x: x[:self.counter_modulo]))
            random_indices = np.random.choice(
                self.max_size if self.filled_max_size else self.counter_modulo,
                BATCH_SIZE
            )
            slicer = lambda x: pre_slicer(x)[random_indices]

            old_observation_neurals = slicer(self.old_observation_neuron_array)
            new_observation_neurals = slicer(self.new_observation_neuron_array)
            action_neurals = slicer(self.action_neuron_array)
            are_not_ends = slicer(self.are_not_end_array)
            rewards = slicer(self.reward_array)
            n_data_points = old_observation_neurals.shape[0]

            prediction = self.predict(
                np.concatenate((old_observation_neurals, new_observation_neurals))
            )
            wip_q_values, new_q_values = np.split(prediction, 2)
            new_other_q_values = self.other_training_data.predict(
                new_observation_neurals
            )

            # Assumes discrete actions:
            action_indices = np.dot(action_neurals, range(n_actions)).astype(np.int32)

            batch_index = np.arange(n_data_points, dtype=np.int32)
            wip_q_values[batch_index, action_indices] = (
                rewards + self.model_free_learning_policy.gamma * are_not_ends *
                new_other_q_values[np.arange(new_q_values.shape[0]),
                                   np.argmax(new_q_values, axis=1)]

            )


            fit_arguments = {
                'x': {name: old_observation_neurals[name] for name
                      in old_observation_neurals.dtype.names},
                'y': wip_q_values,
                'verbose': 0,
            }

            self.model.fit(**fit_arguments)

            self.mark_trained()




    def is_training_time(self) -> bool:
        n_batches = self.counter // self.model_free_learning_policy.training_batch_size
        return n_batches > self._last_trained_batch


    def mark_trained(self) -> None:
        self._last_trained_batch = \
                               self.counter // self.model_free_learning_policy.training_batch_size
        assert not self.is_training_time()

    @property
    def counter_modulo(self) -> int:
        return self.counter % self.max_size

    @property
    def filled_max_size(self) -> bool:
        return self.counter >= self.max_size

    def predict(self, input_array):
        return self.model.predict({name: input_array[name] for name in input_array.dtype.names})





class ModelFreeLearningPolicy(QPolicy):
    def __init__(self, *, epsilon: numbers.Real = 0.1, gamma: numbers.Real = 0.9,
                 training_batch_size: int = 100, n_models: int = 2) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
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


    def get_qs_for_observations(self, observations: Sequence[Observation] = None) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        observation_neurals = [observation.to_neural() for observation in observations]
        if observation_neurals:
            assert utils.is_structured_array(observation_neurals[0])
        input_array = np.concatenate(observation_neurals)
        training_data = random.choice(self.training_datas)
        if training_data.model is None:
            prediction_output = np.random.rand(input_array.shape[0], self.State.Action.n_neurons)
        else:
            prediction_output = training_data.predict(input_array)
        actions = tuple(self.State.Action)
        return tuple(
            {action: q for action, q in dict(zip(actions, output_row)).items()
             if (action in observation.legal_actions)}
            for observation, output_row in zip(observations, prediction_output)
        )



    def get_next_action_and_policy(self, game: Game, reward: numbers.Number,
                                   observation: Observation) -> Tuple[Action, Policy]:
        epsilon = self.epsilon
        if (epsilon > 0) and (epsilon == 1 or epsilon > random.random()):
            # The verbose condition above is an optimized version of `if epsilon > random.random():`
            return random.choice(observation.legal_actions)
        else:
            try:
                q_map = self.q_map_cache[observation]
            except KeyError:
                q_map = self.q_map_cache[observation] = self.get_qs_for_observation(observation)
            return (max(q_map, key=q_map.__getitem__), self)


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

    def create_model(self, observation: Observation, action: Action) -> keras.Model:
        model = keras.models.Sequential(
            layers=(
                keras.layers.Input(
                    shape=observation.to_neural()[0]['sequential'].shape,
                    name='sequential'
                ),
                keras.layers.Dense(
                    256, activation='relu',
                ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    256, activation='relu',
                ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    256, activation='relu',
                ),
                keras.layers.Dropout(rate=0.1),
                keras.layers.Dense(
                    action.to_neural().shape[0] # activation='relu'
                ),
            ),
        )
        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        return model



