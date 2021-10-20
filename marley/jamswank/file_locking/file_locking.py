# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import pathlib
import os
import threading

import filelock

class InternalFileLock(filelock.FileLock):
    pass


class CachedFileLockType(type):
    def __new__(mcls, *args, **kwargs) -> CachedFileLockType:
        result = super().__new__(mcls, *args, **kwargs)
        result.__cache = {}
        return result

    def __call__(cls, path: os.PathLike) -> FileLock:
        path = pathlib.Path(path)
        try:
            return cls.__cache[path]
        except KeyError:
            cls.__cache[path] = file_lock = super().__call__(path)
            return file_lock




class FileLock(metaclass=CachedFileLockType):
    def __init__(self, path: os.PathLike) -> None:
        self.path = pathlib.Path(path)
        self.internal_file_lock = InternalFileLock(path)
        self.threading_lock = threading.RLock()
        self._ensured_folder_exists = False

    def acquire(self) -> None:
        self.threading_lock.acquire()
        if not self._ensured_folder_exists:
            self.path.parent.mkdir(parents=True, exist_ok=True)
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

    def __repr__(self) -> str:
        return f'{type(self).__name__}({repr(self.path)})'


