# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import pathlib
import threading

import filelock

class InternalFileLock(filelock.FileLock):
    pass



class FileLock:
    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.internal_file_lock = InternalFileLock(path)
        self.threading_lock = threading.RLock()

    def acquire(self) -> None:
        self.threading_lock.acquire()
        try:
            self.internal_file_lock.acquire()
        except:
            self.threading_lock.release()
            raise

    def release(self) -> None:
        self.internal_file_lock.release()
        self.threading_lock.release()

    def __enter__(self) -> FileLock:
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.release()



