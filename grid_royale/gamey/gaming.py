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
from .aggregating import Culture, Payoff, State, Activity


class Game:
    def __init__(self, *, states: Iterable[State], cultures: Iterable[Culture],
                 activities: Iterable[Activity], payoffs: Iterable[Payoff]) -> None:
        self.states = list(states)
        self.cultures = list(cultures)
        self.activities = list(activities)
        self.payoffs = list(payoffs)
        self._assert_correct_lengths()

    @classmethod
    def from_state_culture(cls, state: State, culture: Culture) -> Game:
        return cls(states=(state,), cultures=(culture,), activities=(), payoffs=())

    def _assert_correct_lengths(self) -> None:
        assert (len(self.cultures) == len(self.states) ==
                len(self.activities) + 1 == len(self.payoffs) + 1)


    def __iter__(self) -> Iterator[State]:

        self._assert_correct_lengths()

        yield from self.states

        state: State = self.states[-1]
        culture: Culture = self.cultures[-1]

        while not state.is_end:
            activity = culture.get_next_activity(state)
            self.activities.append(activity)

            payoff, state = state.get_next_payoff_and_state(activity)
            self.payoffs.append(payoff)
            self.states.append(state)

            culture = culture.get_next_culture(self.states[-2], activity, payoff, state)
            self.cultures.append(culture)

            self._assert_correct_lengths()
            yield state

    def crunch(self, n: Optional[int] = None) -> None:
        for _ in more_itertools.islice_extended(self)[:n]:
            pass
        assert self.states[-1].is_end or len(self.states) == n

