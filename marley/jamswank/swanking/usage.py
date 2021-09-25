# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional
import itertools

from ..jamming import BaseJamDatabase, Jam, JamId, JamItem, JamKind, JamParchment
from .core import *
from .fields import *


# Usage: (Will be moved out)

class ProgramRun(Swank):
    creation_datetime: datetime_module.datetime


class Experiment(Swank):
    creation_datetime: datetime_module.datetime

class Game(Swank):
    ...


class BlackjackGame(Game):
    policies: Mapping[str, Policy]


class GridRoyaleSwankDatabase(SwankDatabase):
    ...