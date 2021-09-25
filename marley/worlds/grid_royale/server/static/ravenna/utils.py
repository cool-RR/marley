# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import json
import math
import sys
import re
import time
import itertools
import numbers
import collections.abc
import urllib.parse
from typing import Union, Optional, Iterator, Iterable, TypeVar, Callable, Tuple, TypeVar, Sequence
import random

from browser import document, html, ajax, timer, window

def add_parameters_to_url(url: str, parameters: dict) -> str:
    # Todo: This is a caveman implementation, replace with urllib.parse
    if '?' in url:
        raise NotImplementedError
    return f'{url}?{"&".join(f"{key}={value}" for key, value in parameters.items())}'


def cool_ajax(url: str, callback: Callable, method: str = 'GET', *, mode: str = 'text',
              disable_cache: bool = True) -> None:
    assert mode in ('text', 'binary')
    request = ajax.ajax(mode=mode)
    if disable_cache:
        url = add_parameters_to_url(url, {'_': random.randint(0, 10**8)})
    request.open(method, url, True)
    if mode == 'binary':
        request.responseType = 'blob'
    request.bind('complete', callback)
    request.send()


def sliced(seq: Sequence, n: int) -> Iterable[Sequence]:
    """Yield slices of length *n* from the sequence *seq*.

        >>> list(sliced((1, 2, 3, 4, 5, 6), 3))
        [(1, 2, 3), (4, 5, 6)]

    If the length of the sequence is not divisible by the requested slice
    length, the last slice will be shorter.

        >>> list(sliced((1, 2, 3, 4, 5, 6, 7, 8), 3))
        [(1, 2, 3), (4, 5, 6), (7, 8)]

    This function will only work for iterables that support slicing.
    For non-sliceable iterables, see :func:`chunked`.

    """
    # Taken from more_itertools
    return itertools.takewhile(len, (seq[i : i + n] for i in itertools.count(0, n)))
