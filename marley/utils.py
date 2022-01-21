# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''A collection of general-purpose tools.'''

from __future__ import annotations

import tempfile
import shutil
import io
import sys
import numbers
import pathlib
import contextlib
import builtins
import collections
import time as time_module
import threading
import copyreg
import contextlib
from typing import Iterable, Iterator, Hashable, Any, Union, Optional
import time
import datetime as datetime_module

from bidict import bidict
from immutabledict import immutabledict as ImmutableDict




@contextlib.contextmanager
def create_temp_folder(prefix: str = tempfile.template, suffix: str = '',
                       parent_folder: Optional[str] = None,
                       chmod: Optional[str] = None) -> pathlib.Path:
    '''
    Context manager that creates a temporary folder and deletes it after usage.

    After the suite finishes, the temporary folder and all its files and
    subfolders will be deleted.

    Example:

        with create_temp_folder() as temp_folder:

            # We have a temporary folder!
            assert temp_folder.is_dir()

            # We can create files in it:
            (temp_folder / 'my_file').open('w')

        # The suite is finished, now it's all cleaned:
        assert not temp_folder.exists()

    Use the `prefix` and `suffix` string arguments to dictate a prefix and/or a
    suffix to the temporary folder's name in the filesystem.

    If you'd like to set the permissions of the temporary folder, pass them to
    the optional `chmod` argument, like this:

        create_temp_folder(chmod=0o550)

    '''
    temp_folder = pathlib.Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix,
                                                dir=parent_folder))
    try:
        if chmod is not None:
            temp_folder.chmod(chmod)
        yield temp_folder
    finally:
        shutil.rmtree(str(temp_folder))


class SelfLoggingDict(collections.UserDict):

    def __setitem__(self, key: Any, value: Any) -> None:
        collections.UserDict.__setitem__(self, key, value)
        print(f'd[{key}] = {value}')

    def __delitem__(self, key: Any) -> None:
        collections.UserDict.__delitem__(self, key)
        print(f'del d[{key}]')


class SelfLoggingBidict(bidict):
    _fwdm_cls = SelfLoggingDict



class LruDict(collections.OrderedDict):

    def __init__(self, max_size: int, /, *args, **kwargs):
        assert max_size > 0
        self.max_size = max_size

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)

        while len(self) > self.max_size:
            del self[next(iter(self))]

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        finally:
            self.move_to_end(key)

    def peek_last_item(self):
        try:
            last_key = next(reversed(self))
        except StopIteration as stop_iteration:
            raise IndexError from stop_iteration
        return (last_key, self[last_key])
