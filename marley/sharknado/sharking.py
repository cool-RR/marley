# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import os
import time
import threading
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
from types import TracebackType
import itertools
import uuid as uuid_module
import logging

from .weighting import Weight, CalfWeight, TadpoleWeight
from .gaining import ThinGain, FatGain
from .jobbing import Job, ThinJob, FatJob, JobSniffingJob
from .antillesing import Antilles
from .supervising import Supervisor
from .utils import sleep_until, ChangeTracker

logger = logging.getLogger(__name__)



class SharkThread(threading.Thread):
    def __init__(self, shark: Shark) -> None:
        threading.Thread.__init__(self, name=f'shark-{shark.name}')
        self.shark = shark

    def run(self) -> None:
        try:
            self.shark.supervisor.main_loop()
        except BaseException as base_exception:
            logger.exception('Shark thread exited because of an exception.')
            raise SystemExit(1) from base_exception


    def start(self) -> None:
        # This is making sure that the loop is marked as active before it starts, so if we try to
        # request a shutdown before it started, it still makes the request instead of thinking the
        # main loop isn't running.
        self.shark.supervisor.main_loop_active = True
        threading.Thread.start(self)



class Shark:
    def __init__(self, directive_thin_jobs: Union[ThinJob, Iterable[ThinJob]] = (), *,
                 name: Optional[str] = None, n_workers: Optional[int] = None,
                 use_multiprocessing: bool = True, start: bool = False,
                 sniff_only: bool = False, worker_setup_function: Optional[Callable] = None,
                 worker_setup_args: Sequence = ()) -> None:
        self.name = name or uuid_module.uuid4().hex[:6]
        self.shark_thread = None
        self.use_multiprocessing = use_multiprocessing
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.supervisor = Supervisor(
            use_multiprocessing=use_multiprocessing, n_workers=self.n_workers, name=self.name,
            worker_setup_function=worker_setup_function, worker_setup_args=worker_setup_args
        )
        self.antilles = self.supervisor.antilles
        self.job_to_directive_gain = self.antilles.job_to_directive_gain
        self.add_directive_thin_jobs(directive_thin_jobs)
        self.sniff_only = sniff_only
        if start:
            self.start()

    @property
    def sniff_only(self) -> bool:
        return self._sniff_only

    @sniff_only.setter
    def sniff_only(self, value: bool) -> None:
        self.supervisor.sniff_only = self._sniff_only = value

    def start(self) -> None:
        assert self.shark_thread is None
        self.shark_thread = SharkThread(self)
        self.shark_thread.start()

    def add_directive_thin_jobs(self,
                                directive_thin_jobs: Union[ThinJob, Iterable[ThinJob]]) -> None:
        self.supervisor.antilles.job_to_directive_gain.add_thin_jobs(directive_thin_jobs)

    def shut_down(self, *, finish_jobs: bool = True) -> None:
        self.supervisor.request_shut_down(finish_jobs=finish_jobs)
        self.shark_thread.join()
        self.shark_thread = None

    def __enter__(self) -> Shark:
        assert not self.supervisor.requested_shut_down
        if self.shark_thread is None:
            self.start()
        return self

    def __exit__(self, exc_type: Type[Exception], exc_value: Exception,
                 exc_traceback: TracebackType) -> None:
        if self.shark_thread is not None:
            self.shut_down()

    def _sniffs_are_done(self) -> bool:
        return (
            not any(isinstance(job, JobSniffingJob) for job in
                    self.antilles.job_to_available_gain) and
            not self.antilles.job_to_sniff_pending_gain and
            not self.antilles.job_to_pending_gain and
            all((desired_gain in self.antilles.job_to_sniffed_gain[job])
                for job, desired_gain in self.antilles.job_to_desired_gain.items())
        )

    def wait_for_sniffs(self) -> None:
        # todo: This is shit, of course.
        assert self.sniff_only
        assert all(isinstance(job, JobSniffingJob) for job in self.antilles.job_to_pending_gain)
        logger.debug('Waiting for sniffs to finish...')
        change_tracker = ChangeTracker()
        sleep_until(self._sniffs_are_done, 200,
                    reset_condition=lambda: change_tracker(self.antilles._snapshot()))
        logger.debug(f'Done waiting for sniffs to finish. '
                     f'Current state:\n{self.antilles._snapshot().show()}')


