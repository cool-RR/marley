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

from marley.gamey import GameySwankDatabase


class MarleySwankDatabase(GameySwankDatabase):
    def get_swank_types(self) -> dict:
        return super().get_swank_types() | set()



