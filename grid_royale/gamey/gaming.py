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
    def __init__(self, *, cultures: Iterable[Culture], states: Iterable[State],
                 activities: Iterable[Activity], payoffs: Iterable[Payoff]) -> None:
        self.cultures = list(cultures)
        self.states = list(states)
        self.activities = list(activities)
        self.payoffs = list(payoffs)


    @classmethod
    def from_culture_state(cls, culture: Culture, state: State) -> Game:
        return cls(cultures=(culture,), states=(states,), activities=(), payoffs=())



    def __iter__(self) -> Iterator[State]:

        assert (len(self.cultures) == len(self.payoffs) == len(self.states) ==
                len(self.activities) + 1)

        yield from self.states

        culture: Culture = self.cultures[-1]
        payoff: Payoff = self.payoffs[-1]
        state: State = self.states[-1]

        while True:
            culture = culture.get_next_culture(payoff, state)
            self.cultures.append(culture)

            activity = culture.get_next_activity(payoff, state)
            self.activities.append(activity)

            payoff, state = state.get_next_payoff_and_state(activity)
            self.payoffs.append(payoff)
            self.states.append(state)

            yield state

    def crunch(self, n: Optional[int] = None) -> None:
        for _ in more_itertools.islice_extended(self)[:n]:
            pass
        assert self.states[-1].is_end or len(self.states) == n

