# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''Tools for dealing with compatibility with different Python versions.'''

import functools
from typing import Callable,  Any

# Provide functools.cache in older Python versions:
if not hasattr(functools, 'cache'):
    def cache(user_function: Callable, /) -> Any:
        '''Simple lightweight unbounded cache.  Sometimes called "memoize".'''
        return functools.lru_cache(maxsize=None)(user_function)
    functools.cache = cache
