# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import time
import threading
import os
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict, Set)
from types import TracebackType
import concurrent.futures
from bidict import bidict
import logging

import more_itertools

import marley
from . import sharknado
from . import jamswank

def marley_worker_setup_function(swank_database: jamswank.SwankDatabase) -> None:
    from . import gamey
    gamey.global_gamey_swank_database = swank_database

    # Avoid TensorFlow spam:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    logging.getLogger('tensorflow').addFilter(
        lambda record: 'Tracing is expensive and the excessive' not in record.msg
    )




class MarleyShark(sharknado.Shark):
    def __init__(self,
                 directive_thin_jobs: Union[sharknado.ThinJob, Iterable[sharknado.ThinJob]] = (), *,
                 name: Optional[str] = None, use_multiprocessing: bool = True, start: bool = False,
                 sniff_only: bool = False) -> None:
        sharknado.Shark.__init__(
            self, directive_thin_jobs=directive_thin_jobs, name=name,
            use_multiprocessing=use_multiprocessing, start=start, sniff_only=sniff_only,
            worker_setup_function=marley_worker_setup_function,
            worker_setup_args=(marley.constants.swank_database,)
        )
