# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations
from __future__ import generator_stop

from typing import Optional, Union, Tuple, Iterable, Iterator, Sequence

import more_itertools


def iterate_windowed_pairs(iterable: Iterable) -> Iterator:
    iterator = iter(iterable)
    try:
        old = next(iterator)
    except StopIteration:
        return
    for new in iterator:
        yield (old, new)
        old = new


class LastDetectingIterator(more_itertools.peekable):
    def on_last_item(self) -> bool:
        try:
            self.peek()
        except StopIteration:
            return True
        else:
            return False


