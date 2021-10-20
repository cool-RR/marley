# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import click
import webbrowser
import pathlib
import itertools
import re
import time
from typing import (Optional, Tuple, Union, Container, Hashable, Iterator, Mapping,
                    Iterable, Any, Dict, FrozenSet, Callable, Type, Sequence, Set)
import logging
import platform
import sys
import contextlib

from marley.jamswank import jamming
from . import logging_setup
from . import gamey
from .constants import swank_database

logger = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', is_flag=True)
@click.option('--log-to-file/--dont-log-to-file', is_flag=True, default=True)
def marley_command_group(*, verbose: bool = False, log_to_file: bool = True) -> None:
    from marley import __version__
    logging_setup.setup(verbose=verbose, log_to_file=log_to_file)
    logger.debug(f'Starting Marley {__version__}, Python version {platform.python_version()}')
    logger.debug(f'{sys.argv=}')
    gamey.global_gamey_swank_database = swank_database

@marley_command_group.resultcallback()
def marley_done(result: Any, *, verbose: bool = False, log_to_file: bool = True) -> None:
    logger.debug(f'Marley finished, exiting.')

@marley_command_group.command()
def initialize() -> None:
    pass

@click.option('--host', default=None)
@click.option('--port', default=None)
@marley_command_group.command()
def serve(*, host: Optional[str], port: Optional[str]) -> None:
    from . import arezzo
    with arezzo.ArezzoServerThread(host=host, port=port) as arezzo_server_thread:
        logger.info(f'Open {arezzo_server_thread.url} in your browser to access Marley.')
        while True:
            time.sleep(0.1)

####################################################################################################

def is_exit_0_exception(exception: Exception) -> bool:
    return (
        (isinstance(exception, SystemExit) and exception.code == 0) or
        (isinstance(exception, click.exceptions.Exit) and exception.exit_code == 0)
    )


@contextlib.contextmanager
def run_and_log_exception():
    try:
        yield
    except BaseException as base_exception:
        if not is_exit_0_exception(base_exception):
            logger.exception('Marley exited because of an exception.')
            raise SystemExit(1) from base_exception

def import_all_worlds():
    import marley.worlds.grid_royale
    import marley.worlds.blackjack


def marley(*args: Any, **kwargs: Any) -> None:
    # Import all the worlds to register their command groups:
    import_all_worlds()

    with run_and_log_exception():
        marley_command_group(*args, **kwargs)

