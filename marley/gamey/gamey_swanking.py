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

from marley import gamey
from marley.jamswank.swanking import SwankDatabase, Swank
from . import utils


class GameySwank(Swank):
    def __init__(self, *args, swank_database: Optional[SwankDatabase] = None, **kwargs) -> None:
        if swank_database is not None:
            assert swank_database is gamey.global_gamey_swank_database
        Swank.__init__(self, *args, swank_database=gamey.global_gamey_swank_database, **kwargs)


class GameySwankDatabase(SwankDatabase):
    pass





