# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''A collection of general-purpose tools.'''

from __future__ import annotations

import tempfile
import shutil
import io
import sys
import pathlib
import contextlib
import builtins
import time as time_module
import threading
import copyreg
import contextlib
from typing import Iterable, Iterator, Hashable
import time
import datetime as datetime_module


def iterate_deduplicated(iterable: Iterable[Hashable], seen: Iterable[Hashable] = ()) \
                                                                              -> Iterator[Hashable]:
    seen = set(seen)
    for item in iterable:
        if item in seen:
            continue
        else:
            yield item
            seen.add(item)


def pickle_lock(lock):
    return (threading.Lock, ())

copyreg.pickle(type(threading.Lock()), pickle_lock)


def pickle_r_lock(r_lock):
    return (threading.RLock, ())

copyreg.pickle(type(threading.RLock()), pickle_r_lock)


# def pickle_stack_summary(stack_summary):
    # return (tensorflow.compat.v1.flags.tf_decorator.tf_stack.StackSummary, ())


# copyreg.pickle(tensorflow.compat.v1.flags.tf_decorator.tf_stack.StackSummary,
               # pickle_stack_summary)



@contextlib.contextmanager
def BlankContextManager():
    yield

class NotInDict:
    '''Object signifying that the key was not found in the dict.'''


class TempValueSetter(object):
    '''
    Context manager for temporarily setting a value to a variable.

    The value is set to the variable before the suite starts, and gets reset
    back to the old value after the suite finishes.
    '''

    def __init__(self, variable, value, assert_no_fiddling=True):
        '''
        Construct the `TempValueSetter`.

        `variable` may be either an `(object, attribute_string)`, a `(dict,
        key)` pair, or a `(getter, setter)` pair.

        `value` is the temporary value to set to the variable.
        '''

        self.assert_no_fiddling = assert_no_fiddling


        #######################################################################
        # We let the user input either an `(object, attribute_string)`, a
        # `(dict, key)` pair, or a `(getter, setter)` pair. So now it's our job
        # to inspect `variable` and figure out which one of these options the
        # user chose, and then obtain from that a `(getter, setter)` pair that
        # we could use.

        bad_input_exception = Exception(
            '`variable` must be either an `(object, attribute_string)` pair, '
            'a `(dict, key)` pair, or a `(getter, setter)` pair.'
        )

        try:
            first, second = variable
        except Exception:
            raise bad_input_exception
        if hasattr(first, '__getitem__') and hasattr(first, 'get') and \
           hasattr(first, '__setitem__') and hasattr(first, '__delitem__'):
            # `first` is a dictoid; so we were probably handed a `(dict, key)`
            # pair.
            self.getter = lambda: first.get(second, NotInDict)
            self.setter = lambda value: (first.__setitem__(second, value) if
                                         value is not NotInDict else
                                         first.__delitem__(second))
            ### Finished handling the `(dict, key)` case. ###

        elif callable(second):
            # `second` is a callable; so we were probably handed a `(getter,
            # setter)` pair.
            if not callable(first):
                raise bad_input_exception
            self.getter, self.setter = first, second
            ### Finished handling the `(getter, setter)` case. ###
        else:
            # All that's left is the `(object, attribute_string)` case.
            if not isinstance(second, str):
                raise bad_input_exception

            parent, attribute_name = first, second
            self.getter = lambda: getattr(parent, attribute_name)
            self.setter = lambda value: setattr(parent, attribute_name, value)
            ### Finished handling the `(object, attribute_string)` case. ###

        #
        #
        ### Finished obtaining a `(getter, setter)` pair from `variable`. #####


        self.getter = self.getter
        '''Getter for getting the current value of the variable.'''

        self.setter = self.setter
        '''Setter for Setting the the variable's value.'''

        self.value = value
        '''The value to temporarily set to the variable.'''

        self.active = False


    def __enter__(self):

        self.active = True

        self.old_value = self.getter()
        '''The old value of the variable, before entering the suite.'''

        self.setter(self.value)

        # In `__exit__` we'll want to check if anyone changed the value of the
        # variable in the suite, which is unallowed. But we can't compare to
        # `.value`, because sometimes when you set a value to a variable, some
        # mechanism modifies that value for various reasons, resulting in a
        # supposedly equivalent, but not identical, value. For example this
        # happens when you set the current working directory on Mac OS.
        #
        # So here we record the value right after setting, and after any
        # possible processing the system did to it:
        self._value_right_after_setting = self.getter()

        return self


    def __exit__(self, exc_type, exc_value, exc_traceback):

        if self.assert_no_fiddling:
            # Asserting no-one inside the suite changed our variable:
            assert self.getter() == self._value_right_after_setting

        self.setter(self.old_value)

        self.active = False

class OutputCapturer(object):
    '''
    Context manager for catching all system output generated during suite.

    Example:

        with OutputCapturer() as output_capturer:
            print('woo!')

        assert output_capturer.output == 'woo!\n'

    The boolean arguments `stdout` and `stderr` determine, respectively,
    whether the standard-output and the standard-error streams will be
    captured.
    '''
    def __init__(self, stdout=True, stderr=True):
        self.string_io = io.StringIO()

        if stdout:
            self._stdout_temp_setter = \
                TempValueSetter((sys, 'stdout'), self.string_io)
        else: # not stdout
            self._stdout_temp_setter = BlankContextManager()

        if stderr:
            self._stderr_temp_setter = \
                TempValueSetter((sys, 'stderr'), self.string_io)
        else: # not stderr
            self._stderr_temp_setter = BlankContextManager()

    def __enter__(self):
        '''Manage the `OutputCapturer`'s context.'''
        self._stdout_temp_setter.__enter__()
        self._stderr_temp_setter.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # Not doing exception swallowing anywhere here.
        self._stderr_temp_setter.__exit__(exc_type, exc_value, exc_traceback)
        self._stdout_temp_setter.__exit__(exc_type, exc_value, exc_traceback)

    output = property(lambda self: self.string_io.getvalue(),
                      doc='''The string of output that was captured.''')


class TempSysPathAdder(object):
    '''
    Context manager for temporarily adding paths to `sys.path`.

    Removes the path(s) after suite.

    Example:

        with TempSysPathAdder('path/to/fubar/package'):
            import fubar
            fubar.do_stuff()

    '''
    def __init__(self, addition):
        self.addition = [str(addition)]


    def __enter__(self):
        self.entries_not_in_sys_path = [entry for entry in self.addition if
                                        entry not in sys.path]
        sys.path += self.entries_not_in_sys_path
        return self


    def __exit__(self, *args, **kwargs):

        for entry in self.entries_not_in_sys_path:

            # We don't allow anyone to remove it except for us:
            assert entry in sys.path

            sys.path.remove(entry)


def _zip_strict(*iterables):
    if not iterables:
        return
    iterators = tuple(iter(iterable) for iterable in iterables)
    try:
        while True:
            items = []
            for iterator in iterators:
                items.append(next(iterator))
            yield tuple(items)
    except StopIteration:
        if items:
            raise ValueError
        sentinel = object()
        for _, iterator in enumerate(iterators[1:], 1):
            if next(iterator, sentinel) is not sentinel:
                raise ValueError

def zip(*iterables, strict: bool = False):
    if strict is False:
        return builtins.zip(*iterables)
    else:
        return _zip_strict(*iterables)


