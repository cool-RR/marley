# # Copyright 2020 Ram Rachum and collaborators.
# # This program is distributed under the MIT license.

# from __future__ import annotations

# import math
# import inspect
# import re
# import abc
# import random
# import itertools
# import collections.abc
# import statistics
# import concurrent.futures
# import enum
# import functools
# import numbers
# from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    # Sequence, Callable, Hashable, Mapping, TypeVar)
# import dataclasses

# import more_itertools
# import numpy as np

# from .base import State, Activity, Payoff, Culture

# class FStage(abc.ABC):

    # @abc.abstractmethod
    # def get_next_f_stage(self):
        # raise NotImplementedError



# @dataclasses.dataclass(order=True, frozen=True)
# class Fee(FStage):
    # culture: Culture
    # payoff: Payoff
    # state: State

    # def get_fi(self):
        # return Fi(
            # *self,
            # *self.culture.get_next_activity_and_culture(self.payoff, self.state)
        # )

    # get_next_f_stage = get_fi


# @dataclasses.dataclass(order=True, frozen=True)
# class Fi(FStage):
    # culture: Culture
    # payoff: Payoff
    # state: State

    # next_activity: Activity
    # next_culture: Culture

    # def get_fo(self):
        # return Fo(self.state, self.next_activity, self.next_culture)

    # get_next_f_stage = get_fo


# @dataclasses.dataclass(order=True, frozen=True)
# class Fo(FStage):
    # state: State
    # activity: Activity
    # culture: Culture

    # def get_fum(self):
        # return Fum(
            # *self,
            # *self.state.get_next_payoff_and_state(self.activity)
        # )

    # get_next_f_stage = get_fum




# @dataclasses.dataclass(order=True, frozen=True)
# class Fum(FStage):
    # state: State
    # activity: Activity
    # culture: Culture

    # next_payoff: Payoff
    # next_state: State

    # def get_fee(self):
        # return Fee(self.culture, self.payoff, self.state)

    # get_next_f_stage = get_fee



