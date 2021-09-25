# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import time
import multiprocessing.context
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict, Set)
from types import TracebackType
import concurrent.futures
import threading
import queue as queue_module
import atexit

import more_itertools

N_PRE_SPAWN_PROCESSES = 10

waiting_processes = queue_module.Queue()

spawning_thread = None

class SpawningThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self.shutdown_requested = False
        self.start()

    def run(self):
        while not self.shutdown_requested:
            while waiting_processes.qsize() <= N_PRE_SPAWN_PROCESSES:
                process = PreSpawnProcess._create()
                waiting_processes.put(process)
                process.start()
            time.sleep(0.1)


class PreSpawnProcessType(type):

    def _create(cls):
        return type.__call__(cls)

    def __call__(cls, *, target: Callable, args: Iterable,
                 actually_create: bool = False) -> PreSpawnProcess:
        global spawning_thread
        if spawning_thread is None:
            spawning_thread = SpawningThread()
        process = waiting_processes.get()
        process.actually_start(target=target, args=args)
        return process


class PreSpawnProcess(multiprocessing.context.Process, metaclass=PreSpawnProcessType):
    def __init__(self, *args, **kwargs):
        multiprocessing.context.Process.__init__(self, *args, **kwargs)
        self.target_args_queue = multiprocessing.Queue()
        self.lock = multiprocessing.Lock()

    _is_started = False
    # def __init__(self, *, target: Callable, args: Iterable) -> None:
        # multiprocessing.context.Process.__init__(self, args=args)

    def run(self) -> None:
        try:
            target, args = self.target_args_queue.get(timeout=120)
        except queue_module.Empty:
            return
        else:
            target(*args)

    def start(self) -> None:
        with self.lock:
            if not self._is_started:
                multiprocessing.context.Process.start(self)
                self._is_started = True

    def actually_start(self, *, target: Callable, args: Iterable):
        self.target_args_queue.put((target, args))




class PreSpawnContext(multiprocessing.context.DefaultContext):
    Process = PreSpawnProcess

    def __init__(self):
        multiprocessing.context.DefaultContext.__init__(self, multiprocessing.get_context())


class PreSpawnProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    def __init__(self, max_workers=None, *, pre_spawn: bool = False) -> None:
        if pre_spawn:
            mp_context = PreSpawnContext()
        else:
            mp_context = None
        concurrent.futures.ProcessPoolExecutor.__init__(self, max_workers=max_workers,
                                                        mp_context=mp_context)



@atexit.register
def exit():
    if spawning_thread is not None:
        spawning_thread.shutdown_requested = True
    for process in more_itertools.iter_except(waiting_processes.get_nowait, queue_module.Empty):
        process.target_args_queue.put((sum, ((),)))
    time.sleep(0.1)
    for process in more_itertools.iter_except(waiting_processes.get_nowait, queue_module.Empty):
        process.target_args_queue.put((sum, ((),)))



if  __name__ ==  '__main__':
    pre_spawn_process = PreSpawnProcess(target=print, args=(7,))
    pre_spawn_process.start()
    pre_spawn_process.join()
