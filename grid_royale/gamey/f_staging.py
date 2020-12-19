# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import math
import inspect
import re
import abc
import random
import itertools
import collections.abc
import statistics
import concurrent.futures
import enum
import functools
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import dataclasses

import more_itertools
import numpy as np

from .base import State, Action, Activity
from .strategizing import Mind


class FStage:
    pass

@dataclasses.dataclass(order=True, frozen=True)
class Fee(FStage):
    state: State
    Activity
    def __init__(self, ):
