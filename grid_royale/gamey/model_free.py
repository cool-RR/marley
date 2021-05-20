# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import random
import weakref
import concurrent.futures
import numbers
import functools
import collections.abc
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Mapping, MutableMapping)
import hashlib
import weakref
import dataclasses
import pathlib
import tempfile

import keras.models
import more_itertools
import numpy as np
import lru # todo: probably replace

from grid_royale import gamey
from .base import Observation, Action, Story
from .policing import Policy, QPolicy
from . import utils
from .timelining import Timeline, StoryDoesntFitInTimeline

DEFAULT_MAX_BATCH_SIZE = 64
DEFAULT_MAX_PAST_MEMORY_SIZE = 1_000

class MustDefineCustomModel(NotImplementedError):
    pass


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    create_model: Callable
    observation_neural_dtype: np.dtype
    action_n_neurons: int

    def __call__(self) -> keras.Model:
        return self.create_model(self.observation_neural_dtype, self.action_n_neurons)


def _serialize_model(model: keras.Model) -> bytes:
    with tempfile.TemporaryDirectory() as temp_folder:
        path = pathlib.Path(temp_folder) / 'model.h5'
        model.save_weights(path, save_format='h5')
        return path.read_bytes()

def _deserialize_to_model(model: keras.Model,
                          serialized_model: collections.abc.ByteString) -> None:
    with tempfile.TemporaryDirectory() as temp_folder:
        path = pathlib.Path(temp_folder) / 'model.h5'
        path.write_bytes(serialized_model)
        model.load_weights(path)




class ModelJockey(collections.abc.Mapping):
    def __init__(self, model_spec: ModelSpec, max_size: int = 5) -> None:
        self.model_spec = model_spec
        self.weak_model_tracker = weakref.WeakValueDictionary()
        self.model_to_serialized_model = utils.WeakKeyIdentityDict()
        self.serialized_model_to_model = lru.LRU(max_size)

    @property
    def max_size(self) -> int:
        return self.serialized_model_to_model.get_size()

    def __getitem__(self, serialized_model: bytes) -> keras.Model:
        try:
            return self.serialized_model_to_model[serialized_model]
        except KeyError:
            model = self.model_spec()
            _deserialize_to_model(model, serialized_model)

            self.weak_model_tracker[id(model)] = model
            self.model_to_serialized_model[model] = serialized_model
            self.serialized_model_to_model[serialized_model] = model
            return model


    def get_random_model(self) -> keras.Model:
        model = self.model_spec()
        serialized_model = _serialize_model(model)

        self.weak_model_tracker[id(model)] = model
        self.model_to_serialized_model[model] = serialized_model
        self.serialized_model_to_model[serialized_model] = model
        return model


    def get_random_serialized_model(self) -> keras.Model:
        return self.model_to_serialized_model[self.get_random_model()]

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.serialized_model_to_model)

    def __len__(self) -> int:
        return len(self.serialized_model_to_model)


    def serialize_model(self, model: keras.Model) -> bytes:
        try:
            return self.model_to_serialized_model[model]
        except AttributeError:
            self.model_to_serialized_model[model] = serialized_model = _serialize_model(model)
            return serialized_model


    def clone_model_and_train(self, serialized_model: bytes,
                              train: Callable[[keras.Model], None]) -> bytes:
        if len(self) < self.max_size:
            model = self.model_spec()
        else:
            # Todo: Two race condition: Peek and get, and someone using this model elsewhere
            oldest_serialized_model, model = self.serialized_model_to_model.peek_last_item()
            del self.serialized_model_to_model[oldest_serialized_model]
            del self.model_to_serialized_model[model]

        _deserialize_to_model(model, serialized_model)
        train(model)
        new_serialized_model = _serialize_model(model)

        self.weak_model_tracker[id(model)] = model
        self.model_to_serialized_model[model] = new_serialized_model
        self.serialized_model_to_model[new_serialized_model] = model

        return new_serialized_model



class ModelMegaJockey(collections.abc.Mapping):

    def __init__(self):
        self._dict = {}

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._dict)

    def __getitem__(self, model_spec: ModelSpec) -> ModelJockey:
        try:
            return self._dict[model_spec]
        except KeyError:
            self._dict[model_spec] = model_jockey = ModelJockey(model_spec)
            return model_jockey


model_mega_jockey = ModelMegaJockey()

class ModelManager(collections.abc.Sequence):
    def __init__(self, model_free_learning_policy: ModelFreeLearningPolicy) -> None:
        self.model_free_learning_policy = model_free_learning_policy

    def __len__(self):
        return len(self.model_free_learning_policy.serialized_models)

    def __getitem__(self, i: Union[int, slice]) -> keras.Model:
        if isinstance(i, slice):
            raise NotImplementedError
        assert isinstance(i, int)
        return self.model_free_learning_policy.model_jockey[
            self.model_free_learning_policy.serialized_models[i]
        ]




class ModelFreeLearningPolicy(QPolicy):
    def __init__(self, *, action_type: Optional[Type[Action]] = None,
                 observation: Optional[Union[Observation, Type[Observation]]] = None,
                 observation_neural_dtype: Optional[np.dtype] = None,
                 serialized_models: Optional[Sequence[bytes]] = None,
                 epsilon: numbers.Real = 0.1, discount: numbers.Real = 0.9,
                 training_counter: int = 0, training_period: numbers.Real = 100, n_models: int = 2,
                 timelines: Iterable[Timeline] = ()) -> None:
        if action_type is None:
            assert self.Action is not None
        else:
            self.Action = action_type

        if observation_neural_dtype is not None:
            assert observation is None
            self.observation_neural_dtype = observation_neural_dtype
        elif observation is not None:
            self.observation_neural_dtype = observation.neural_dtype
        elif hasattr(self, 'Observation'):
            self.observation_neural_dtype = self.Observation.neural_dtype
        else:
            assert self.observation_neural_dtype is not None # Defined as class attribute

        self.epsilon = epsilon
        self.discount = discount
        self.training_counter = training_counter
        self.training_period = training_period
        if serialized_models is None:
            self.serialized_models = tuple(self.model_jockey.get_random_serialized_model()
                                           for _ in range(n_models))
        else:
            assert len(serialized_models) == n_models
            self.serialized_models = tuple(serialized_models)

        self.timelines = tuple(timelines)
        self.fingerprint = hashlib.sha512(b''.join(self.serialized_models)).hexdigest()[:6]
        self.models = ModelManager(self)


    @property
    def _model_kwargs(self):
        return dict(observation_neural_dtype=self.observation_neural_dtype,
                    action_n_neurons=self.Action.n_neurons)

    @property
    def _model_spec(self):
        return ModelSpec(self.create_model, **self._model_kwargs)

    def _get_clone_kwargs(self):
        return {
            'epsilon': self.epsilon,
            'discount': self.discount,
            'training_counter': self.training_counter,
            'training_period': self.training_period,
            'serialized_models': self.serialized_models,
            'n_models': len(self.models),
            'timelines': self.timelines,
            'action_type': self.Action,
            'observation_neural_dtype': self.observation_neural_dtype,
        }

    def _get_qs_for_observations_uncached(self, observations: Sequence[Observation] = None) -> \
                                                               Tuple[Mapping[Action, numbers.Real]]:
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

    def train(self, games: Sequence[gamey.Game]) -> ModelFreeLearningPolicy:
        clone_kwargs = self._get_clone_kwargs()

        timelines = list(self.timelines)
        for timeline in reversed(timelines):
            try:
                timeline += story
            except StoryDoesntFitInTimeline:
                # This story belongs on a different timeline, keep searching for it:
                continue
            else:
                # We found the right timeline and added the story to it, we're done:
                break
        else:
            # No matching timelines, start a new one:
            timelines.append(Timeline.make_initial(story))

        clone_kwargs['timelines'] = tuple(timelines)

        if self.training_counter + 1 == self.training_period: # It's training time!
            clone_kwargs['training_counter'] == 0
            clone_kwargs['serialized_models'] = self._train_model_and_cycle()

        else:  # It's not training time.
            clone_kwargs['training_counter'] += 1
        return type(self)(**clone_kwargs)


    def clone_and_train(self, n: int = 1, *, max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                        max_past_memory_size: int = DEFAULT_MAX_PAST_MEMORY_SIZE) -> \
                                                                            ModelFreeLearningPolicy:
        clone_kwargs = self._get_clone_kwargs()

        clone_kwargs['training_counter'] = 0

        for _ in range(n):
            clone_kwargs['serialized_models'] = self._train_model_and_cycle(
                clone_kwargs['serialized_models'],
                max_batch_size=max_batch_size,
                max_past_memory_size=max_past_memory_size,
            )

        return type(self)(**clone_kwargs)


    def clone_without_timelines(self) -> ModelFreeLearningPolicy:
        clone_kwargs = {
            **self._get_clone_kwargs(),
            'timelines': (),
        }
        return type(self)(**clone_kwargs)



    def _train_model_and_cycle(self, serialized_models: Optional[Tuple[bytes]] = None,
                               max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                               max_past_memory_size: int = DEFAULT_MAX_PAST_MEMORY_SIZE) -> \
                                                                                       Tuple[bytes]:
        serialized_models = (self.serialized_models if serialized_models is None
                             else serialized_models)
        train_model = lambda model: self._train_model(
            model, other_model=self.model_jockey[serialized_models[-1]],
            max_batch_size=max_batch_size, max_past_memory_size=max_past_memory_size
        )
        serialized_trained_model = self.model_jockey.clone_model_and_train(
            serialized_models[0], train_model
        )
        return serialized_models[1:] + (serialized_trained_model,)



    def get_next_action(self, observation: Observation) -> Action:
        epsilon = self.epsilon
        if (epsilon > 0) and (epsilon == 1 or epsilon > random.random()):
            # The verbose condition above is an optimized version of `if epsilon > random.random():`
            return random.choice(observation.legal_actions)
        else:
            q_map = self.get_qs_for_observation(observation) # This is cached.
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

    @property
    def model_jockey(self) -> ModelJockey:
        return model_mega_jockey[self._model_spec]


    @staticmethod
    def create_model(observation_neural_dtype: np.dtype, action_n_neurons: int) -> keras.Model:
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

        return model


    def predict(self, model: keras.Model, input_array: np.ndarray) -> np.ndarray:
        return model.predict({name: input_array[name] for name in input_array.dtype.names})


    def _train_model(self, model: keras.Model, *, other_model: keras.Model,
                     max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                     max_past_memory_size: int = DEFAULT_MAX_PAST_MEMORY_SIZE) -> None:

        ### Getting a random selection of stories to train on: #####################################
        #                                                                                          #
        past_memory = utils.ChainSpace(map(reversed, reversed(self.timelines)))
        foo = min(max_past_memory_size, len(past_memory))
        batch_size = min(max_batch_size, foo)
        indices = tuple(sorted(random.sample(range(foo), batch_size)))
        stories = tuple(past_memory[index] for index in indices)
        #                                                                                          #
        ### Finished getting a random selection of stories to train on. ############################

        ### Initializing arrays: ###################################################################
        #                                                                                          #
        old_observation_neural_array = np.zeros(
            (batch_size,), dtype=self.observation_neural_dtype
        )
        action_neural_array = np.zeros(
            (batch_size, self.Action.n_neurons), dtype=bool
        )
        reward_array = np.zeros(batch_size)
        new_observation_neural_array = np.zeros(
            (batch_size,), dtype=self.observation_neural_dtype
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
            reward_array + self.discount * are_not_end_array *
            new_other_q_values[np.arange(new_q_values.shape[0]),
                               np.argmax(new_q_values, axis=1)]
        )


        fit_arguments = {
            'x': {name: old_observation_neural_array[name] for name
                  in old_observation_neural_array.dtype.names},
            'y': wip_q_values,
            'verbose': 0,
            'epochs': 10,
        }

        model.fit(**fit_arguments)
