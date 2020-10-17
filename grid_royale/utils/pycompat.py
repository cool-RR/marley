# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import functools

if not hasattr(functools, 'cache'):
    def cache(user_function, /):
        '''Simple lightweight unbounded cache.  Sometimes called "memoize".'''
        return functools.lru_cache(maxsize=None)(user_function)
    functools.cache = cache