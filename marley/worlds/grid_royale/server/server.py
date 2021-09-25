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


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 22390
STATIC_FOLDER = pathlib.Path(__file__).parent / 'static'

assert STATIC_FOLDER.exists()
assert STATIC_FOLDER.is_dir()

def make_url_from_host_port(host: str, port: Union[str, int]):
    return urllib.parse.urlunsplit(('http', f'{host}:{port}', '', '', ''))

DEFAULT_URL = make_url_from_host_port(DEFAULT_HOST, DEFAULT_PORT)


app = flask.Flask('ravenna', static_url_path='/static', static_folder=str(STATIC_FOLDER))
jamming_server.add_to_flask_app(constants.jam_file_database, app)

@app.route('/')
def index_redirect():
    return flask.redirect('/static/index.html')


@app.route('/shutdown', methods=('POST',))
def shutdown() -> str:
    shutdown_function = flask.request.environ.get('werkzeug.server.shutdown')
    if shutdown_function is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_function()
    return 'Shut down flask server.'

@app.route('/games')
def games() -> str:
    return json.dumps([path.name for path in get_games_folder().iterdir()
                       if path.is_dir() and any(path.iterdir())])

@app.route('/games/<game>/<chunk>.json')
def game_chunk(game: str, chunk: str) -> str:
    assert re.fullmatch('^[0-9]+$', chunk)
    try:
        return (get_games_folder() / game / f'{chunk}.json').read_text()
    except FileNotFoundError:
        raise flask.abort(404)

def _shut_up_flask():
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.ERROR)
    app.logger.disabled = logger.disabled = True
    flask.cli.show_server_banner = lambda *args, **kwargs: None
    warnings.filterwarnings(
        'ignore',
        message="The 'environ['werkzeug.server.shutdown']' function is deprecated"
    )


class ServerThread(threading.Thread):
    def __init__(self, host: str = DEFAULT_HOST, port: Union[int, str] = DEFAULT_PORT, *,
                 quiet: bool = False) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.host = host
        self.port = int(port)
        self.quiet = quiet

    def run(self) -> None:
        if self.quiet:
            _shut_up_flask()
        app.run(host=self.host, port=self.port)

    def shutdown(self) -> None:
        response = requests.post(urllib.parse.urljoin(DEFAULT_URL, 'shutdown'))
        response.raise_for_status()
        self.join()

    def __enter__(self) -> ServerThread:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.shutdown()

    @property
    def url(self):
        host = '127.0.0.1' if self.host == '0.0.0.0' else self.host
        return make_url_from_host_port(host, self.port)


