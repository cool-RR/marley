# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import functools
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

from grid_royale.base import games_folder


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 22390

def make_url_from_host_port(host: str, port: Union[str, int]):
    return urllib.parse.urlunsplit(('http', f'{host}:{port}', '', '', ''))

DEFAULT_URL = make_url_from_host_port(DEFAULT_HOST, DEFAULT_PORT)


app = flask.Flask('grid_royale')

frontend_folder = pathlib.Path(__file__).parent / 'frontend'

@functools.lru_cache()
def read_file(file_name) -> str:
    assert file_name in ('grid_royale.html', 'grid_royale.js', 'grid_royale.py')
    return (frontend_folder / file_name).read_text()

@functools.lru_cache()
def read_file_binary(file_name) -> str:
    assert file_name in ('play.png', 'pause.png')
    return (frontend_folder / file_name).read_bytes()


### Defining views for simple static files: ###################################
#                                                                             #
@app.route('/')
def index() -> str:
    return read_file('grid_royale.html')

@app.route('/grid_royale.py')
def grid_royale_py() -> str:
    return read_file('grid_royale.py')

@app.route('/grid_royale.js')
def grid_royale_js() -> str:
    return read_file('grid_royale.js')

@app.route('/play.png')
def play_png() -> str:
    return read_file_binary('play.png')

@app.route('/pause.png')
def pause_png() -> str:
    return read_file_binary('pause.png')
#                                                                             #
### Finished defining views for simple static files. ##########################


@app.route('/shutdown', methods=('POST',))
def shutdown() -> str:
    shutdown_function = flask.request.environ.get('werkzeug.server.shutdown')
    if shutdown_function is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_function()
    return 'Shut down flask server.'

@app.route('/games')
def games() -> str:
    return json.dumps([path.name for path in games_folder.iterdir()
                       if path.is_dir() and any(path.iterdir())])

@app.route('/games/<game>/<chunk>.json')
def game_chunk(game: str, chunk: str) -> str:
    assert re.fullmatch('^[0-9]+$', chunk)
    try:
        return (games_folder / game / f'{chunk}.json').read_text()
    except FileNotFoundError:
        raise flask.abort(404)

def _shut_up_flask():
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.ERROR)
    app.logger.disabled = logger.disabled = True
    flask.cli.show_server_banner = lambda *args, **kwargs: None


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


