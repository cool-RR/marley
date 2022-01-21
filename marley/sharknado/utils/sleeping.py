# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import operator as operator_module
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import collections.abc
import numbers
import itertools
import functools
import time as time_module

from marley import utils


def sleep_until(condition: Callable[[], Any], total_seconds: float = 10, *,
                step_seconds: float = 0.5, initial_seconds: float = 0,
                reset_condition: Optional[Callable[[], Any]] = None,
                exception: Union[Exception, Type[Exception]] = TimeoutError) -> None:
    if initial_seconds:
        time_module.sleep(initial_seconds)
    if condition():
        return
    div, mod = divmod(total_seconds - initial_seconds, step_seconds)
    n_steps = int(div + int(bool(mod)))
    while True:
        for _ in range(n_steps):
            time_module.sleep(step_seconds)
            if condition():
                return
            elif reset_condition is not None and reset_condition():
                break # to `while` loop, restarting the wait
        else:
            raise exception


