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
import flask_cors

from marley import constants
from marley.worlds.grid_royale.core import get_games_folder
from marley.jamswank.jamming import server as jamming_server
from marley.jamswank.jamming.jam_file_database import JamFileDatabase
from marley.jamswank.swanking import EmptyJam

from .user_sessioning import ArezzoUserSession


DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 22390
DIST_FOLDER = pathlib.Path(__file__).parent / 'arezzo' / 'dist'

assert DIST_FOLDER.exists()
assert DIST_FOLDER.is_dir()

def make_url_from_host_port(host: str, port: Union[str, int]):
    return urllib.parse.urlunsplit(('http', f'{host}:{port}', '', '', ''))

DEFAULT_URL = make_url_from_host_port(DEFAULT_HOST, DEFAULT_PORT)


app = flask.Flask('arezzo', static_url_path='/', static_folder=str(DIST_FOLDER))
flask_cors.CORS(app)
jamming_server.add_to_flask_app(constants.jam_file_database, app)

@app.errorhandler(404)
def page_not_found(exception) -> tuple[str, int]:
    return ((DIST_FOLDER / 'index.html').read_text(), 200)



@app.route('/api/shutdown', methods=('POST',))
def shutdown() -> str:
    shutdown_function = flask.request.environ.get('werkzeug.server.shutdown')
    if shutdown_function is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    shutdown_function()
    return 'Shut down flask server.'

@app.route('/api/favorites', methods=('GET',))
def get_favorites():
    try:
        arezzo_user_session = ArezzoUserSession.load(constants.swank_database)
    except IndexError:
        return json.dumps([])
    else:
        return json.dumps(arezzo_user_session.favorites)

@app.route('/api/favorites/<path:favorite>', methods=('POST',))
def add_favorite(favorite):
    if not re.fullmatch(r'^[a-zA-Z0-9._/\[\]*]+$', favorite):
        return flask.Response(status=400)
    with ArezzoUserSession.lock_and_load_or_create(constants.swank_database,
                                                   save=True) as arezzo_user_session:
        arezzo_user_session.favorites = sorted(set(arezzo_user_session.favorites) |
                                               {favorite})
    return ''

@app.route('/api/favorites/<path:favorite>', methods=('DELETE',))
def delete_favorite(favorite):
    if not re.fullmatch(r'^[a-zA-Z0-9._/\[\]*]+$', favorite):
        return flask.Response(status=400)
    with ArezzoUserSession.lock_and_load_or_create(constants.swank_database,
                                                   save=True) as arezzo_user_session:
        arezzo_user_session.favorites = sorted(set(arezzo_user_session.favorites) -
                                               {favorite})
    return ''


def _shut_up_flask():
    logger = logging.getLogger('werkzeug')
    logger.setLevel(logging.ERROR)
    app.logger.disabled = logger.disabled = True
    flask.cli.show_server_banner = lambda *args, **kwargs: None
    warnings.filterwarnings(
        'ignore',
        message="The 'environ['werkzeug.server.shutdown']' function is deprecated"
    )


class ArezzoServerThread(threading.Thread):
    def __init__(self, host: Optional[str] = DEFAULT_HOST,
                 port: Union[int, str, None] = DEFAULT_PORT, *,
                 quiet: bool = False) -> None:
        threading.Thread.__init__(self, daemon=True)
        self.host = host if (host is not None) else DEFAULT_HOST
        self.port = int(port) if (port is not None) else DEFAULT_PORT
        self.quiet = quiet

    def run(self) -> None:
        if self.quiet:
            _shut_up_flask()
        app.run(host=self.host, port=self.port)

    def shutdown(self) -> None:
        response = requests.post(urllib.parse.urljoin(DEFAULT_URL, 'shutdown'))
        response.raise_for_status()
        self.join()

    def __enter__(self) -> ArezzoServerThread:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.shutdown()

    @property
    def url(self):
        host = '127.0.0.1' if self.host == '0.0.0.0' else self.host
        return make_url_from_host_port(host, self.port)


