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

from .base import State, Action, Activity, Payoff
from .strategizing import Policy

class FStage:
    pass

@dataclasses.dataclass(order=True, frozen=True)
class Fee(FStage):
    state: State
    activity: Activity
    culture: Culture

    def get_fi(self):
        return Fi(
            *self,
            *self.state.get_next_payoff_and_state(self.activity)
        )

    get_next_f_stage = get_fi


@dataclasses.dataclass(order=True, frozen=True)
class Fi(FStage):
    old_state: State
    activity: Activity
    culture: Culture

    payoff: Payoff
    state: State

    def get_fo(self):
        return Fo(self.culture, self.payoff, self.state)

    get_next_f_stage = get_fo


@dataclasses.dataclass(order=True, frozen=True)
class Fo(FStage):
    culture: Culture
    payoff: Payoff
    state: State

    def get_fum(self):
        return Fi(
            *self,
            *self.culture.get_next_activity_and_culture(self.payoff, self.state)
        )

    get_next_f_stage = get_fum




@dataclasses.dataclass(order=True, frozen=True)
class Fum(FStage):
    old_culture: Culture
    payoff: Payoff
    state: State

    activity: Activity
    culture: Culture

    def get_fee(self):
        return Fee(self.state, self.activity, self.culture)

    get_next_f_stage = get_fee



