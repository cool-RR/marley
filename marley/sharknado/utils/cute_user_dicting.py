# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import collections


class CuteUserDict(collections.UserDict):

    def __repr__(self) -> str:
        return f'{type(self).__name__}({collections.UserDict.__repr__(self)})'



