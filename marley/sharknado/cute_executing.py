# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import time
import threading
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict, Set)
from types import TracebackType
import concurrent.futures
from bidict import bidict
import logging

import more_itertools

from .weighting import Weight, CalfWeight, TadpoleWeight
from .gaining import ThinGain, FatGain
from .jobbing import Job, ThinJob, FatJob
from .antillesing import Antilles
from . import utils

logger = logging.getLogger(__name__)


DEFAULT_MAX_FUTURES = 100

class CuteExecutorFull(Exception):
    pass

def worker_setup_then_function(worker_setup_function: Optional[callable],
                               worker_setup_args: Sequence, function: Callable) -> Any:
    if worker_setup_function is not None:
        worker_setup_function(*worker_setup_args)
    return function()



class CuteExecutorMixin(concurrent.futures.Executor):
    def __init__(self, *, max_futures: int = DEFAULT_MAX_FUTURES,
                 worker_setup_function: Optional[Callable] = None,
                 worker_setup_args: Sequence = ()) -> None:
        self.max_futures = max_futures
        self.futures = set()
        self.worker_setup_function = worker_setup_function
        self.worker_setup_args = worker_setup_args

    def can_accept_more_futures(self) -> bool:
        return len(self.futures) < self.max_futures

    def submit(self, function: Callable) -> concurrent.futures.Future:
        if not self.can_accept_more_futures():
            raise CuteExecutorFull
        else:
            future = super().submit(worker_setup_then_function, self.worker_setup_function,
                                    self.worker_setup_args, function)
            self.futures.add(future)
            return future

    def _get_completed_futures(self) -> Set[concurrent.futures.Future]:
        iterator = concurrent.futures.as_completed(self.futures, timeout=0)
        return set(
            more_itertools.iter_except(
                iterator.__next__,
                (StopIteration, concurrent.futures.TimeoutError)
            )
        )

    def get_completed_futures(self, *, wait_for_at_least_one_if_non_empty: bool = False) -> \
                                                                     Set[concurrent.futures.Future]:
        completed_futures = self._get_completed_futures()
        if wait_for_at_least_one_if_non_empty and not completed_futures and self.futures:
            logger.debug(f"There are no completed futures, and there are {len(self.futures)} "
                         f"pending futures. We were asked to wait in this scenario for just one "
                         f"future, so we'll wait.")
            completed_futures = {more_itertools.first(
                                                     concurrent.futures.as_completed(self.futures))}

        for future in completed_futures:
            future: concurrent.futures.Future
            future.result() # If the future got an exception, raise it here.
            self.futures.remove(future)
        return completed_futures

    # def iterate_futures_as_completed(self) -> Iterator[concurrent.futures.Future]:
        # for future in concurrent.futures.as_completed(self.futures):
            # self.futures.remove(future)
            # yield future


class CuteProcessExecutor(CuteExecutorMixin, concurrent.futures.ProcessPoolExecutor):
    def __init__(self, max_workers=None, *, max_futures: int = DEFAULT_MAX_FUTURES,
                 worker_setup_function: Optional[Callable] = None,
                 worker_setup_args: Sequence = ()) -> None:
        CuteExecutorMixin.__init__(self, max_futures=max_futures,
                                   worker_setup_function=worker_setup_function,
                                   worker_setup_args=worker_setup_args)
        concurrent.futures.ProcessPoolExecutor.__init__(self, max_workers=max_workers)
        for _ in range(2 * max_workers):
            concurrent.futures.ProcessPoolExecutor.submit(self, sum, (),)

class CuteThreadExecutor(CuteExecutorMixin, concurrent.futures.ThreadPoolExecutor):
    def __init__(self, max_workers=None, *, max_futures: int = DEFAULT_MAX_FUTURES,
                 worker_setup_function: Optional[Callable] = None,
                 worker_setup_args: Sequence = ()) -> None:
        CuteExecutorMixin.__init__(self, max_futures=max_futures,
                                   worker_setup_function=worker_setup_function,
                                   worker_setup_args=worker_setup_args)
        concurrent.futures.ThreadPoolExecutor.__init__(self, max_workers=max_workers)
