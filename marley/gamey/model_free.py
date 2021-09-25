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

import more_itertools
import numpy as np

import marley.utils
from marley import gamey
from marley.jamswank.swanking import SimpleField, SavvyField, JamId, SwankDatabase
from .base import Observation, Action, Story
from .policing import Policy, QPolicy
from . import utils

DEFAULT_MAX_BATCH_SIZE = 128
DEFAULT_MAX_PAST_MEMORY_SIZE = 1_000

class MustDefineCustomModel(NotImplementedError):
    pass


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    create_model: Callable
    observation_neural_dtype: np.dtype
    action_n_neurons: int

    def __call__(self) -> 'keras.Model':
        return self.create_model(self.observation_neural_dtype, self.action_n_neurons)


def _serialize_model(model: 'keras.Model') -> bytes:
    with marley.utils.create_temp_folder() as temp_folder:
        path = pathlib.Path(temp_folder) / 'model.h5'
        model.save_weights(path, save_format='h5')
        return path.read_bytes()

def _deserialize_to_model(model: 'keras.Model',
                          serialized_model: collections.abc.ByteString) -> None:
    with marley.utils.create_temp_folder() as temp_folder:
        path = pathlib.Path(temp_folder) / 'model.h5'
        path.write_bytes(serialized_model)
        model.load_weights(path)




class ModelJockey(collections.abc.Mapping):
    def __init__(self, model_spec: ModelSpec, max_size: int = 5) -> None:
        self.model_spec = model_spec
        self.weak_model_tracker = weakref.WeakValueDictionary()
        self.model_to_serialized_model = utils.WeakKeyIdentityDict()
        self.serialized_model_to_model = marley.utils.LruDict(max_size)

    @property
    def max_size(self) -> int:
        return self.serialized_model_to_model.max_size

    def __getitem__(self, serialized_model: bytes) -> 'keras.Model':
        try:
            return self.serialized_model_to_model[serialized_model]
        except KeyError:
            model = self.model_spec()
            _deserialize_to_model(model, serialized_model)

            self.weak_model_tracker[id(model)] = model
            self.model_to_serialized_model[model] = serialized_model
            self.serialized_model_to_model[serialized_model] = model
            return model


    def get_random_model(self) -> 'keras.Model':
        model = self.model_spec()
        serialized_model = _serialize_model(model)

        self.weak_model_tracker[id(model)] = model
        self.model_to_serialized_model[model] = serialized_model
        self.serialized_model_to_model[serialized_model] = model
        return model


    def get_random_serialized_model(self) -> 'keras.Model':
        return self.model_to_serialized_model[self.get_random_model()]

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.serialized_model_to_model)

    def __len__(self) -> int:
        return len(self.serialized_model_to_model)


    def serialize_model(self, model: 'keras.Model') -> bytes:
        try:
            return self.model_to_serialized_model[model]
        except AttributeError:
            self.model_to_serialized_model[model] = serialized_model = _serialize_model(model)
            return serialized_model


    def clone_model_and_train(self, serialized_model: bytes,
                              train: Callable[['keras.Model'], None]) -> bytes:
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

    def __getitem__(self, i: Union[int, slice]) -> 'keras.Model':
        if isinstance(i, slice):
            raise NotImplementedError
        assert isinstance(i, int)
        return self.model_free_learning_policy.model_jockey[
            self.model_free_learning_policy.serialized_models[i]
        ]




class ModelFreeLearningPolicy(QPolicy):

    action_type: Type[Action] = SavvyField()
    observation_neural_dtype: np.dtype = SavvyField()
    epsilon: numbers.Real = SimpleField()
    discount: numbers.Real = SimpleField()
    serialized_models: Sequence[bytes] = SavvyField()

    @classmethod
    def create(cls, *, action_type: Optional[Type[Action]] = None,
               observation: Optional[Union[Observation, Type[Observation]]] = None,
               observation_neural_dtype: Optional[np.dtype] = None,
               serialized_models: Optional[Sequence[bytes]] = None, default_n_models: int = 2,
               epsilon: numbers.Real = 0.1, discount: numbers.Real = 0.9) -> None:


        ### Figuring out `action_type`: ############################################################
        #                                                                                          #
        if action_type is None:
            default_action_type = cls.action_type.default_factory()
            assert issubclass(default_action_type, Action)
            action_type = default_action_type
        #                                                                                          #
        ### Finished figuring out `action_dtype`. ##################################################


        ### Figuring out `observation_neural_dtype`: ###############################################
        #                                                                                          #
        if observation_neural_dtype is not None:
            assert observation is None
        elif observation is not None:
            observation_neural_dtype = observation.neural_dtype
        elif hasattr(cls, 'observation_type'):
            observation_neural_dtype = cls.observation_type.neural_dtype
        else:
            assert isinstance(cls.observation_neural_dtype, np.dtype) is not None
            # Defined as class attribute
        #                                                                                          #
        ### Finished figuring out `observation_neural_dtype`. ######################################

        ### Figuring out `serialized_models`: ######################################################
        #                                                                                          #
        if serialized_models is None:
            model_jockey = model_mega_jockey[
                ModelSpec(
                    cls.create_model, observation_neural_dtype=observation_neural_dtype,
                    action_n_neurons=action_type.n_neurons
                )
            ]
            serialized_models = tuple(model_jockey.get_random_serialized_model()
                                      for _ in range(default_n_models))
        else:
            serialized_models = tuple(serialized_models)
        #                                                                                          #
        ### Finished figuring out `serialized_models`: #############################################

        return cls(
            action_type=action_type, observation_neural_dtype=observation_neural_dtype,
            epsilon=epsilon, discount=discount, serialized_models=serialized_models
        )



    def __init__(self, *, action_type: Type[Action], observation_neural_dtype: np.dtype,
                 epsilon: numbers.Real, discount: numbers.Real,
                 serialized_models: Sequence[bytes], jam_id: Optional[JamId] = None,
                 jam_index: Optional[int] = None,
                 swank_database: Optional[SwankDatabase] = None) -> None:
        QPolicy.__init__(
            self, action_type=action_type, observation_neural_dtype=observation_neural_dtype,
            epsilon=epsilon, discount=discount, serialized_models=serialized_models, jam_id=jam_id,
            jam_index=jam_index, swank_database=swank_database

        )
        self.fingerprint = hashlib.sha512(b''.join(self.serialized_models)).hexdigest()[:6]
        self.models = ModelManager(self)


    @property
    def _model_kwargs(self):
        return dict(observation_neural_dtype=self.observation_neural_dtype,
                    action_n_neurons=self.action_type.n_neurons)

    @property
    def _model_spec(self):
        return ModelSpec(self.create_model, **self._model_kwargs)

    def _get_qs_for_observations_uncached(self, observations: Sequence[Observation] = None) -> \
                                                               Tuple[Mapping[Action, numbers.Real]]:
        if not observations:
            return ()
        observation_neurals = [observation.neural for observation in observations]
        if observation_neurals:
            assert utils.is_structured_array(observation_neurals[0])
        input_array = np.concatenate(observation_neurals)
        model = random.choice(self.models)
        prediction_output = self.predict(model, input_array)
        actions = tuple(self.action_type)
        return tuple(
            {action: q for action, q in dict(zip(actions, output_row)).items()
             if (action in observation.legal_actions)}
            for observation, output_row in zip(observations, prediction_output)
        )

    def train(self, narratives: Sequence[gamey.Narrative], *, n_epochs: int = 10,
              max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
              max_past_memory_size: int = DEFAULT_MAX_PAST_MEMORY_SIZE) -> ModelFreeLearningPolicy:

        clone_kwargs = self._Swank__field_values

        serialized_models = self.serialized_models
        train_model = lambda model: self._train_model(
            model, narratives, n_epochs=n_epochs,
            other_model=self.model_jockey[serialized_models[-1]],
            max_batch_size=max_batch_size, max_past_memory_size=max_past_memory_size
        )
        serialized_trained_model = self.model_jockey.clone_model_and_train(
            serialized_models[0], train_model
        )
        clone_kwargs['serialized_models'] = serialized_models[1:] + (serialized_trained_model,)
        return type(self)(**clone_kwargs)



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
    def create_model(observation_neural_dtype: np.dtype, action_n_neurons: int) -> 'keras.Model':
        from tensorflow import keras
        if tuple(observation_neural_dtype.fields) != ('sequential_input',):
            raise MustDefineCustomModel
        model = keras.models.Sequential(
            layers=(
                keras.layers.Input(
                    shape=observation_neural_dtype['sequential_input'].shape,
                    name='sequential_input'
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


    def predict(self, model: 'keras.Model', input_array: np.ndarray) -> np.ndarray:
        return model.predict({name: input_array[name] for name in input_array.dtype.names})


    def _train_model(self, model: 'keras.Model', narratives: Sequence[gamey.Narrative], *,
                     other_model: 'keras.Model', n_epochs: int,
                     max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
                     max_past_memory_size: int = DEFAULT_MAX_PAST_MEMORY_SIZE) -> None:

        ### Getting a random selection of stories to train on: #####################################
        #                                                                                          #
        past_memory = utils.ChainSpace(map(reversed, reversed(narratives)))
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
            (batch_size, self.action_type.n_neurons), dtype=bool
        )
        reward_array = np.zeros(batch_size)
        new_observation_neural_array = np.zeros(
            (batch_size,), dtype=self.observation_neural_dtype
        )
        are_not_end_array = np.zeros(batch_size, dtype=bool)

        for i, story in enumerate(stories):
            story: Story
            old_observation_neural_array[i] = story.old_observation.neural
            action_neural_array[i] = story.action.to_neural()
            reward_array[i] = story.reward
            new_observation_neural_array[i] = story.new_observation.neural
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


        action_indices = np.dot(action_neural_array,
                                range(self.action_type.n_neurons)).astype(np.int32)
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
            'epochs': n_epochs,
        }

        model.fit(**fit_arguments)
