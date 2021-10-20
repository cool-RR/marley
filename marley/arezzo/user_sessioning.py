# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import functools
import warnings
import re
import json
import urllib.parse
import threading
import pathlib
import requests
import logging
from typing import (Optional, Tuple, Union, Container, Hashable, Iterator,
                    Iterable, Any, Dict, FrozenSet, Callable, Type, Sequence)

import flask

from marley import constants
from marley.worlds.grid_royale.core import get_games_folder
from marley.jamswank.jamming import server as jamming_server
from marley.jamswank.jamming.jam_file_database import JamFileDatabase
from marley.jamswank import SingletonSwank, SimpleField


class ArezzoUserSession(SingletonSwank):
    favorites = SimpleField(lambda: [])
