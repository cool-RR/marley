# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import concurrent.futures
import numbers
import functools
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable)
import weakref

import keras.models
import more_itertools
import numpy as np
from python_toolbox.combi import ChainSpace # Must remove this dependency

from .base import Observation, Action, Story
from .policing import Policy, QPolicy
from . import utils
from .timelining import Timeline

BATCH_SIZE = 64
MAX_PAST_MEMORY_SIZE = 1_000

class MustDefineCustomModel(NotImplementedError):
    pass



class ModelFreeLearningPolicy(QPolicy):
    def __init__(self, *, observation_neural_dtype: np.dtype, action_n_neurons: int,
                 serialized_models: Optional[Sequence[bytes]], epsilon: numbers.Real = 0.1,
                 gamma: numbers.Real = 0.9, training_counter: int = 0,
                 training_batch_size: int = 100, n_models: int = 2) -> None:
        self.observation_neural_dtype = observation_neural_dtype
        self.action_n_neurons = action_n_neurons
        self.epsilon = epsilon
        self.gamma = gamma
        self.training_counter = training_counter
        self.training_batch_size = training_batch_size
        if serialized_models is None:
            self.models = tuple(self.create_model() for _ in range(n_models))
        else:
            assert len(serialized_models) == n_models
            self.models = tuple(self.create_model(serialized_model) for serialized_model
                                in serialized_models)


        self.training_datas = tuple(TrainingData(self) for _ in range(n_models))
        for left_training_data, right_training_data in utils.iterate_windowed_pairs(
                                                   self.training_datas + (self.training_datas[0],)):
            left_training_data.other_training_data = right_training_data

        self.q_map_cache = weakref.WeakKeyDictionary()
        self.timelines: Tuple[Timeline] = ()



    def train(self, observation: Observation, action: Action,
              nxxxxxext_observation: Observation) -> None:

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
            prediction_output = np.random.rand(input_array.shape[0], self.action_n_neurons)
        else:
            prediction_output = training_data.predict(input_array)
        actions = tuple(self.State.Action)
        return tuple(
            {action: q for action, q in dict(zip(actions, output_row)).items()
             if (action in observation.legal_actions)}
            for observation, output_row in zip(observations, prediction_output)
        )


    def get_clone_kwargs(self):
        return {
            'observation_neural_dtype': self.observation_neural_dtype,
            'action_n_neurons': self.action_n_neurons,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'training_counter': self.training_counter,
            'training_batch_size': self.training_batch_size,
            'serialized_models': tuple(model.save_weights() for model in self.models),
            'n_models': len(self.models),
            'timelines': self.timelines,
        }



    def get_next_policy(self, story: Story) -> Policy:
        clone_kwargs = self.get_clone_kwargs()

        if self.training_counter + 1 == self.training_batch_size: # It's training time!
            clone_kwargs['training_counter'] == 0
            serialized_models = []
            for model in self.models:
                cloned_model = keras.models.clone_model(model)
                self._train_model(cloned_model)
                self.model_cache[(weights := model.get_weights(), self.observation_neural_dtype,
                                  self.action_n_neurons)] = cloned_model
                serialized_models.append(weights)

            clone_kwargs['serialized_models'] = tuple(serialized_models)
            clone_kwargs['training_counter'] == 0
        else:  # It's not training time.
            clone_kwargs['training_counter'] += 1
        return type(self)(**clone_kwargs,)

    def get_next_action(self, observation: Observation) -> Action:
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

    model_cache = {}

    @staticmethod
    def create_model(observation_neural_dtype: np.dtype,
                     action_n_neurons: int,
                     serialized_model: Optional[bytes] = None) -> keras.Model:
        if tuple(observation_neural_dtype.fields) != ('sequential',):
            raise MustDefineCustomModel
        model = keras.models.Sequential(
            layers=(
                keras.layers.Input(
                    shape=observation_neural_dtype['sequential'].shape,
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
                    action_n_neurons # activation='relu'
                ),
            ),
        )
        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        if serialized_model is not None:
            model.load_weights(serialized_model)

        return model


    def get_or_create_model(self, serialized_model: Optional[bytes] = None) -> keras.Model:
        if serialized_model is None:
            return self.create_model(self.observation_neural_dtype, self.action_n_neurons,
                                     None)
        key = (self.create_model, self.observation_neural_dtype, self.action_n_neurons,
               serialized_model)
        try:
            return self.model_cache[key]
        except KeyError:
            self.model_cache[key] = self.create_model(self.observation_neural_dtype,
                                                      self.action_n_neurons,
                                                      serialized_model)
            return self.model_cache[key]


    def predict(self, model: keras.Model, input_array: np.ndarray) -> np.ndarray:
        return model.predict({name: input_array[name] for name in input_array.dtype.names})


    def _train_model(self, model: keras.Model):

        ### Getting a random selection of stories to train on: ################
        #                                                                     #
        past_memory = ChainSpace(map(reversed, reversed(self.timelines)))
        indices = utils.random_ints_in_range(0, MAX_PAST_MEMORY_SIZE, BATCH_SIZE)
        stories = tuple(past_memory[index] for index in indices)
        #                                                                     #
        ### Finished getting a random selection of stories to train on. #######

        ### Initializing arrays: ##############################################
        #                                                                     #
        old_observation_neural_array = np.zeros(
            (BATCH_SIZE,), dtype=self.observation_neural_dtype
        )
        action_neural_array = np.zeros(
            (BATCH_SIZE, self.action_n_neurons), dtype=bool
        )
        reward_array = np.zeros(BATCH_SIZE)
        new_observation_neural_array = np.zeros(
            (BATCH_SIZE,), dtype=self.observation_neural_dtype
        )
        are_not_end_array = np.zeros(BATCH_SIZE, dtype=bool)

        for i, story in enumerate(stories):
            story: Story
            old_observation_neural_array[i] = story.old_observation.to_neural()
            action_neural_array[i] = story.action.to_neural()
            reward_array[i] = story.reward
            new_observation_neural_array[i] = story.new_observation.to_neural()
            are_not_end_array[i] = not story.new_observation.state.is_end

        #                                                                     #
        ### Finished initializing arrays. #####################################

        n_actions = len(self.model_free_learning_policy.State.Action)

        prediction = self.predict(
            np.concatenate((old_observation_neural_array, new_observation_neural_array))
        )
        wip_q_values, new_q_values = np.split(prediction, 2)
        new_other_q_values = self.other_training_data.predict(
            new_observation_neural_array
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
            'x': {name: old_observation_neural_array[name] for name
                  in old_observation_neural_array.dtype.names},
            'y': wip_q_values,
            'verbose': 0,
        }

        model.fit(**fit_arguments)
