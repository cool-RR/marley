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

from .utils import ImmutableDict
from . import utils
from . import exceptions
from .base import Culture, Payoff, State, Activity


class Game:
    def __init__(self, culture: Culture, state: State, payoff: Optional[Payoff] = None) -> None:
        self.cultures = [culture]
        self.payoffs = [payoff if (payoff is not None) else Payoff.make_zero(culture)]
        self.states = [state]
        self.activities = []


    def __iter__(self) -> Iterator[State]:
        yield from self.states

        culture: Culture = self.cultures[-1]
        payoff: Payoff = self.payoffs[-1]
        state: State = self.states[-1]

        while not state.is_end:
            activity, culture = culture.get_next_activity_and_culture(self, payoff, state)
            self.activities.append(activity)
            self.cultures.append(culture)

            payoff, state = state.get_next_payoff_and_state(activity)
            self.payoffs.append(payoff)
            self.states.append(state)

            yield state

