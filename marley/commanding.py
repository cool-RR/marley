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

from marley.jamswank import jamming
from . import logging_setup
from . import gamey
from .constants import swank_database

logger = logging.getLogger(__name__)


@click.group()
@click.option('-v', '--verbose', is_flag=True)
@click.option('--log-to-file/--dont-log-to-file', is_flag=True, default=True)
def marley(*, verbose: bool = False, log_to_file: bool = True) -> None:
    from marley import __version__
    logging_setup.setup(verbose=verbose, log_to_file=log_to_file)
    logger.debug(f'Starting Marley {__version__}, Python version {platform.python_version()}')
    logger.debug(f'{sys.argv=}')
    gamey.global_gamey_swank_database = swank_database

@marley.resultcallback()
def marley_done(result: Any, *, verbose: bool = False, log_to_file: bool = True) -> None:
    logger.debug(f'Marley finished, exiting.')

@marley.command()
def initialize() -> None:
    pass

