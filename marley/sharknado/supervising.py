# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import os
import time
import threading
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict, Set)
from types import TracebackType
import concurrent.futures
from bidict import bidict
import uuid as uuid_module
import logging

import more_itertools

import marley.utils
from .weighting import Weight, CalfWeight, TadpoleWeight
from .gaining import ThinGain, FatGain, GainDyad
from .jobbing import Job, ThinJob, FatJob
from .antillesing import Antilles
from .cute_executing import CuteThreadExecutor, CuteProcessExecutor

logger = logging.getLogger(__name__)




class Supervisor:
    def __init__(self, *, use_multiprocessing: bool = True, n_workers: Optional[int],
                 sniff_only: bool = False, name: Optional[str] = None,
                 worker_setup_function: Optional[Callable] = None,
                 worker_setup_args: Sequence = ()) -> None:
        self.name = name or uuid_module.uuid4().hex[:6]
        self.antilles = Antilles(name=self.name)
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.requested_shut_down_when_finished_tasks = self.requested_shut_down = \
                                                                       self.main_loop_active = False
        self.future_to_pending_gain = bidict() # marley.utils.SelfLoggingBidict()
        self.pending_gain_to_future = self.future_to_pending_gain.inverse
        self.cute_executor = None
        self.sniff_only = sniff_only
        self.worker_setup_function = worker_setup_function
        self.worker_setup_args = worker_setup_args

    def get_executor_type(self) -> Type:
        return CuteProcessExecutor if self.use_multiprocessing else CuteThreadExecutor


    def main_loop(self) -> None:
        self.main_loop_active = True
        CuteExecutor = self.get_executor_type()
        with CuteExecutor(self.n_workers, worker_setup_function=self.worker_setup_function,
                          worker_setup_args=self.worker_setup_args) as self.cute_executor:
            while not self.requested_shut_down:
                doable_available_gains_iterator = self.antilles. \
                         iterate_doable_available_gains_and_mark_pending(sniff_only=self.sniff_only)
                while self.cute_executor.can_accept_more_futures():
                    logger.debug('Asking Antilles for a doable gain...')
                    try:
                        doable_available_gain = next(doable_available_gains_iterator)
                    except StopIteration:
                        logger.debug('Antilles is out of doable gains.')
                        break
                    else:
                        logger.debug(f'Antilles gave us this doable gain that we will now '
                                     f'schedule for execution: {doable_available_gain}')
                        future = self.cute_executor.submit(doable_available_gain)
                        future.gain = doable_available_gain # Todo: Maybe remove
                        assert doable_available_gain not in self.pending_gain_to_future
                        self.pending_gain_to_future[doable_available_gain] = future
                        logger.debug(f"Now that we scheduled that doable gain, there are "
                                     f"{len(self.pending_gain_to_future)} pending gains "
                                     f"that we're waiting for.")

                finished_gain_dyads = []
                for future in self.cute_executor.get_completed_futures(
                                                           wait_for_at_least_one_if_non_empty=True):
                    finished_gain_dyads.append(
                        GainDyad(
                            requested_gain=self.future_to_pending_gain.pop(future),
                            returned_gain=future.result()
                        )
                    )
                if finished_gain_dyads:
                    logger.debug(f'{len(finished_gain_dyads)} gains finished running, '
                                 f'reporting to Antilles.')
                    self.antilles.report_finished_gain_dyads(finished_gain_dyads)
                else:
                    if (self.requested_shut_down_when_finished_tasks and
                                                                    not self.cute_executor.futures):
                        break
                    else:
                        time.sleep(0.1) # That'll do, pig. That'll do.
        self.requested_shut_down_when_finished_tasks = self.requested_shut_down = \
                                                                       self.main_loop_active = False

    def request_shut_down(self, *, finish_jobs: bool = True) -> None:
        if self.main_loop_active:
            if finish_jobs:
                self.requested_shut_down_when_finished_tasks = True
            else:
                self.requested_shut_down = True