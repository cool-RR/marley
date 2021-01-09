# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import concurrent.futures
import numbers
import functools
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable)
import hashlib
import weakref

import keras.models
import more_itertools
import numpy as np
from python_toolbox.combi import ChainSpace # Must remove this dependency

from .base import Observation, Action, Story
from .policing import Policy, QPolicy
from . import utils
from .timelining import Timeline, StoryDoesntFitInTimeline

MAX_BATCH_SIZE = 64
MAX_PAST_MEMORY_SIZE = 1_000

class MustDefineCustomModel(NotImplementedError):
    pass



class ModelFreeLearningPolicy(QPolicy):
    Observation: Type[Observation]
    Action: Type[Action]
    def __init__(self, *, serialized_models: Optional[Sequence[bytes]] = None,
                 epsilon: numbers.Real = 0.1, gamma: numbers.Real = 0.9, training_counter: int = 0,
                 training_batch_size: int = 100, n_models: int = 2,
                 timelines: Iterable[Timeline] = ()) -> None:
        self.epsilon = epsilon
        self.gamma = gamma
        self.training_counter = training_counter
        self.training_batch_size = training_batch_size
        if serialized_models is None:
            self.models = tuple(self.get_or_create_model() for _ in range(n_models))
            self.serialized_models = tuple(utils.keras_model_weights_to_bytes(model)
                                           for model in self.models)
        else:
            assert len(serialized_models) == n_models
            self.serialized_models = serialized_models
            self.models = tuple(self.get_or_create_model(serialized_model) for serialized_model
                                in serialized_models)

        self.q_map_cache = weakref.WeakKeyDictionary()
        self.timelines = tuple(timelines)
        self.fingerprint = hashlib.sha512(b''.join(self.serialized_models)).hexdigest()[:6]

    @property
    def _model_kwargs(self):
        return dict(observation_neural_dtype=self.Observation.neural_dtype,
                    action_n_neurons=self.Action.n_neurons)

    @property
    def _clone_kwargs(self):
        return {
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'training_counter': self.training_counter,
            'training_batch_size': self.training_batch_size,
            'serialized_models': self.serialized_models,
            'n_models': len(self.models),
            'timelines': self.timelines,
        }

    def get_qs_for_observations(self, observations: Sequence[Observation] = None) \
                                                            -> Tuple[Mapping[Action, numbers.Real]]:
        observation_neurals = [observation.to_neural() for observation in observations]
        if observation_neurals:
            assert utils.is_structured_array(observation_neurals[0])
        input_array = np.concatenate(observation_neurals)
        model = random.choice(self.models)
        prediction_output = self.predict(model, input_array)
        actions = tuple(self.Action)
        return tuple(
            {action: q for action, q in dict(zip(actions, output_row)).items()
             if (action in observation.legal_actions)}
            for observation, output_row in zip(observations, prediction_output)
        )


    def get_next_policy(self, story: Story) -> Policy:
        clone_kwargs = self._clone_kwargs

        timelines = list(self.timelines)
        try:
            timelines[-1] += story
        except StoryDoesntFitInTimeline:
            timelines.append(Timeline.make_initial(story))
        except IndexError:
            if timelines:
                raise
            timelines.append(Timeline.make_initial(story))

        clone_kwargs['timelines'] = tuple(timelines)

        if self.training_counter + 1 == self.training_batch_size: # It's training time!
            clone_kwargs['training_counter'] == 0

            models = self.clone_model_and_train_one()
            clone_kwargs['serialized_models'] = tuple(
                utils.keras_model_weights_to_bytes(model) for model in models
            )

        else:  # It's not training time.
            clone_kwargs['training_counter'] += 1
        return type(self)(**clone_kwargs)


    def clone_model_and_train_one(self) -> Tuple[keras.Model]:
        models = list(self.models)
        random_index = random.randint(0, len(models) - 1)
        other_index = (random_index + 1) % len(models)
        # (It's the index of another model if there's >=2, otherwise it's the same model.)
        models[random_index] = cloned_model = self.create_model(
            **self._model_kwargs,
            serialized_model=self.serialized_models[random_index]
        )
        self._train_model(cloned_model, other_model=models[other_index])
        self.model_cache[(self.create_model, *self._model_kwargs.values(),
                          utils.keras_model_weights_to_bytes(cloned_model))] = cloned_model
        return tuple(models)



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

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: {self.fingerprint}>'

    model_cache = {}

    @staticmethod
    def create_model(observation_neural_dtype: np.dtype, action_n_neurons: int,
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
        if serialized_model is None:
            utils.keras_model_weights_to_bytes(model) # Save to cache
        else:
            utils.load_keras_model_weights_from_bytes(model, serialized_model,
                                                      save_to_cache=False)

        return model


    def get_or_create_model(self, serialized_model: Optional[bytes] = None) -> keras.Model:
        if serialized_model is None:
            model = self.create_model(**self._model_kwargs)
            key = (self.create_model, *self._model_kwargs.values(),
                   utils.keras_model_weights_to_bytes(model))
            self.model_cache[key] = model
            return model
        else:
            key = (self.create_model, *self._model_kwargs.values(), serialized_model)
            try:
                return self.model_cache[key]
            except KeyError:
                self.model_cache[key] = self.create_model(**self._model_kwargs,
                                                          serialized_model=serialized_model)
                return self.model_cache[key]


    def predict(self, model: keras.Model, input_array: np.ndarray) -> np.ndarray:
        return model.predict({name: input_array[name] for name in input_array.dtype.names})


    def _train_model(self, model: keras.Model, *, other_model: keras.Model) -> None:

        ### Getting a random selection of stories to train on: #####################################
        #                                                                                          #
        past_memory = ChainSpace(map(reversed, reversed(self.timelines)))
        foo = min(MAX_PAST_MEMORY_SIZE, len(past_memory))
        batch_size = min(MAX_BATCH_SIZE, foo)
        indices = utils.random_ints_in_range(0, foo, batch_size)
        stories = tuple(past_memory[index] for index in indices)
        #                                                                                          #
        ### Finished getting a random selection of stories to train on. ############################

        ### Initializing arrays: ###################################################################
        #                                                                                          #
        old_observation_neural_array = np.zeros(
            (batch_size,), dtype=self.Observation.neural_dtype
        )
        action_neural_array = np.zeros(
            (batch_size, self.Action.n_neurons), dtype=bool
        )
        reward_array = np.zeros(batch_size)
        new_observation_neural_array = np.zeros(
            (batch_size,), dtype=self.Observation.neural_dtype
        )
        are_not_end_array = np.zeros(batch_size, dtype=bool)

        for i, story in enumerate(stories):
            story: Story
            old_observation_neural_array[i] = story.old_observation.to_neural()
            action_neural_array[i] = story.action.to_neural()
            reward_array[i] = story.reward
            new_observation_neural_array[i] = story.new_observation.to_neural()
            are_not_end_array[i] = not story.new_observation.state.is_end

        #                                                                                          #
        ### Finished initializing arrays. ##########################################################

        prediction = self.predict(
            model,
            np.concatenate((old_observation_neural_array, new_observation_neural_array))
        )
        wip_q_values, new_q_values = np.split(prediction, 2)

        if other_model is model:
            new_other_q_values = new_q_values
        else:
            new_other_q_values = self.predict(
                other_model,
                new_observation_neural_array
            )


        action_indices = np.dot(action_neural_array, range(self.Action.n_neurons)).astype(np.int32)
        batch_index = np.arange(batch_size, dtype=np.int32)
        wip_q_values[batch_index, action_indices] = (
            reward_array + self.gamma * are_not_end_array *
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
