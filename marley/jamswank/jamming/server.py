# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import re
import io
import abc
import itertools
import bisect
import pathlib
import collections.abc
from typing import (Union, Iterable, Any, Mapping, TypeVar, Optional, Tuple, Sequence, Sized,
                    Callable)
import datetime as datetime_module
import string
import random
import flask
import json

from . import utils
from . import jamming

routes = {}
flask_app_name_to_jam_database = {}


def add_to_routes(path: str) -> Callable:
    def inner(view_function: Callable) -> Callable:
        assert path.startswith('/')
        assert path not in routes
        routes[path] = view_function
        return view_function
    return inner

def _get_jam_database_for_current_app() -> jamming.BaseJamDatabase:
    return flask_app_name_to_jam_database[flask.current_app.name]

@add_to_routes('/')
def jamming_index() -> str:
    return 'yee fucking haw bitch'

@add_to_routes('/<jam_kind_name>/')
def jamming_get_jam_parchment_names(jam_kind_name: str) -> str:
    assert re.fullmatch('[a-z_]+', jam_kind_name)
    jam_database = _get_jam_database_for_current_app()
    return json.dumps(tuple(str(jam_parchment.jam_id) for jam_parchment in
                            jam_database[jam_kind_name]))


@add_to_routes('/<jam_kind_name>/<jam_id_name>/<int:start>..<int:end>/')
def jamming_read_texts(jam_kind_name: str, *, jam_id_name: str, start: int,
                       end: Optional[int]) -> str:
    jam_database = _get_jam_database_for_current_app()
    assert re.fullmatch('[a-z_]+', jam_kind_name)
    jam_id = jamming.JamId(jam_id_name)
    jam_parchment: jamming.JamParchment = jam_database[jam_kind_name][jam_id]
    return ''.join(jam_parchment.read_texts(start, end))


def add_to_flask_app(jam_database: jamming.BaseJamDatabase, app: flask.Flask,
                     prefix: str = '/jamming') -> None:
    assert not prefix.endswith('/')
    assert app.name not in flask_app_name_to_jam_database
    flask_app_name_to_jam_database[app.name] = jam_database
    assert re.fullmatch('^/[a-z0-9_]+$', prefix)
    for path, view_function in routes.items():
        full_path = utils.removesuffix(prefix + path, '/')
        app.add_url_rule(full_path, view_func=view_function)


