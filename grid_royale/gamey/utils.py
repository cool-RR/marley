# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import generator_stop

import numpy as np
import pathlib
import keras
import collections
import dataclasses
from typing import Optional, Union, Tuple, Iterable, Iterator, Sequence
import numbers
import random
import tempfile

import more_itertools
from immutabledict import immutabledict as ImmutableDict


class LastDetectingIterator(more_itertools.peekable):
    def on_last_item(self):
        try:
            self.peek()
        except StopIteration:
            return True
        else:
            return False


def cute_div(x: numbers.Number, y: numbers.Number) -> numbers.Number:
    '''Divide x by y, allowing to divide by 0 and giving infinity with the right sign.'''
    try:
        return x / y
    except ZeroDivisionError:
        if x == 0:
            raise
        else:
            return x * float('inf')

def clamp(number, /, minimum, maximum):
    if number < minimum:
        return minimum
    elif number > maximum:
        return maximum
    else:
        return number


def iterate_windowed_pairs(iterable: Iterable) -> Iterator:
    iterator = iter(iterable)
    try:
        old = next(iterator)
    except StopIteration:
        return
    for new in iterator:
        yield (old, new)
        old = new


class NiceDataclass(collections.abc.Sequence):
    __len__ = lambda self: len(dataclasses.fields(self))
    __iter__ = lambda self: map(
        self.__dict__.__getitem__,
        (field.name for field in dataclasses.fields(self))
    )
    __getitem__ = lambda self, i: tuple(self)[i]

def is_structured_array(array: np.ndarray) -> bool:
    return isinstance(array.dtype, np.dtype) and len(array.dtype) >= 1

def shuffled(sequence: Sequence) -> list:
    '''
    Return a list with all the items from `sequence` shuffled.

    Example:

        >>> random_tools.shuffled([0, 1, 2, 3, 4, 5])
        [0, 3, 5, 1, 4, 2]

    '''
    sequence_copy = list(sequence)
    random.shuffle(sequence_copy)
    return sequence_copy

def random_ints_in_range(start: int, stop: int, n: int) -> Tuple[int]:
    if n > (stop - start):
        raise ValueError
    result = set()
    while len(result) < n:
        result.add(random.randint(start, stop - 1))
    return tuple(sorted(result))


def keras_model_weights_to_bytes(model: keras.Model) -> bytes:
    try:
        return model._cached_serialized_model
    except AttributeError:
        with tempfile.TemporaryDirectory() as temp_folder:
            path = pathlib.Path(temp_folder) / 'model.h5'
            model.save_weights(path, save_format='h5')
            model._cached_serialized_model = path.read_bytes()
            return model._cached_serialized_model

def load_keras_model_weights_from_bytes(model: keras.Model,
                                        weights: collections.abc.ByteString, *,
                                        save_to_cache: bool = True) -> None:
    with tempfile.TemporaryDirectory() as temp_folder:
        path = pathlib.Path(temp_folder) / 'model.h5'
        path.write_bytes(weights)
        model.load_weights(path)
        if save_to_cache:
            model._cached_serialized_model = weights
