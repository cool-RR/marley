# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from typing import Iterable, Sequence, Iterable
import itertools


def iterable_length(iterable: Iterable) -> int:
    for i, _ in enumerate(iterable):
        pass
    return i + 1


def removesuffix(self: str, suffix: str, /) -> str:
    # Backport of `str.removesuffix` from Python 3.9.
    if suffix and self.endswith(suffix):
        return self[:-len(suffix)]
    else:
        return self[:]


def sliced(seq: Sequence, n: int) -> Iterable[Sequence]:
    """Yield slices of length *n* from the sequence *seq*.

        >>> list(sliced((1, 2, 3, 4, 5, 6), 3))
        [(1, 2, 3), (4, 5, 6)]

    If the length of the sequence is not divisible by the requested slice
    length, the last slice will be shorter.

        >>> list(sliced((1, 2, 3, 4, 5, 6, 7, 8), 3))
        [(1, 2, 3), (4, 5, 6), (7, 8)]

    This function will only work for iterables that support slicing.
    For non-sliceable iterables, see :func:`chunked`.

    """
    # Taken from more_itertools
    return itertools.takewhile(len, (seq[i : i + n] for i in itertools.count(0, n)))
