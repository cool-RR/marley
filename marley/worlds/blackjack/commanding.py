# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import click
import webbrowser
import pathlib
import re
import time
from typing import (Optional, Tuple, Union, Container, Hashable, Iterator, Mapping,
                    Iterable, Any, Dict, FrozenSet, Callable, Type, Sequence, Set)
import logging
import platform
import sys

from marley import sharknado
import marley.commanding
import marley.sharknadoing
from .sharknadoing import *

logger = logging.getLogger(__name__)


@marley.commanding.marley_command_group.group(name='blackjack')
def command_group() -> None:
    pass

@click.option('-g', '--n-generations', default=3)
@click.option('-a', '--n-agents', default=2)
@click.option('--use-multiprocessing/--dont-use-multiprocessing', is_flag=True, default=True)
# @click.option('--n-evaluation-games', default=3_000)
@command_group.command()
def run(*, n_generations: int, n_agents: int, use_multiprocessing: bool) -> None:
    # blackjack_project = BlackjackProject()
    # blackjack_project.save()
    blackjack_project = BlackjackProject.get_last(marley.constants.swank_database)
    with marley.sharknadoing.MarleyShark(use_multiprocessing=use_multiprocessing) as shark:
        shark.add_directive_thin_jobs(blackjack_project.get_job(n_generations=n_generations,
                                                                n_agents=n_agents))


@command_group.command()
def write_chart() -> None:
    blackjack_project: BlackjackProject = BlackjackProject.get_last(marley.constants.swank_database)
    blackjack_project.write_chart_to_desktop()