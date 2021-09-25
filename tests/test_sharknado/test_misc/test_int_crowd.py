# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import itertools

from marley.sharknado.int_crowding import IntCrowd, Interval, Point
import more_itertools


def test_one_dimensional():
    empty = IntCrowd()
    i0 = IntCrowd(Interval[2:9])
    i1 = IntCrowd((Interval[2:5], Interval[5:9]))
    i2 = IntCrowd((Interval[2:5], Interval[3:9]))
    i3 = IntCrowd(Interval[3:12])
    i4 = IntCrowd(Interval[30:100])

    assert i0 == i1 == i2 != i3 != i4
    assert repr(i0) == repr(i1) == repr(i2) != repr(i3) != repr(i4)
    assert repr(i0) == repr(i1) == repr(i2) == 'IntCrowd(Interval[2:9])'
    assert repr(i3) == 'IntCrowd(Interval[3:12])'
    assert repr(i4) == 'IntCrowd(Interval[30:100])'
    assert repr(empty) == 'IntCrowd()'

    assert i0
    assert i3
    assert i4
    assert not empty

    assert i0 == (i0 | i0) == (i0 | i0 | i0) == (i1 | i2 | i0)
    assert empty == (i0 - i0) == (i2 - i2) == (i3 - i3)

    assert (i0 | i4 - i4) == i0
    assert (i4 | i0 - i0) == i4

    assert tuple(empty) == ()
    assert len(tuple(i0)) == 7
    assert len(tuple(i4)) == 70



def test_three_dimensional():
    empty = IntCrowd()
    i0 = IntCrowd(Interval[7, 2:9, 5])
    i1 = IntCrowd((Interval[7, 2:5, 5], Interval[7, 5:9, 5]))
    i2 = IntCrowd((Interval[7, 2:9, 5], Interval[7, 3:9, 5]))
    i3 = IntCrowd(Interval[7, 3:12, 5])
    i4 = IntCrowd(Interval[7, 30:100, 5])

    assert i0 == i1 == i2 != i3 != i4
    assert repr(i0) == repr(i1) == repr(i2) != repr(i3) != repr(i4)
    assert repr(i0) == repr(i1) == repr(i2) == 'IntCrowd(Interval[7, 2:9, 5])'
    assert repr(i3) == 'IntCrowd(Interval[7, 3:12, 5])'
    assert repr(i4) == 'IntCrowd(Interval[7, 30:100, 5])'
    assert repr(empty) == 'IntCrowd()'

    assert i0
    assert i3
    assert i4
    assert not empty

    assert i0 == (i0 | i0) == (i0 | i0 | i0) == (i1 | i2 | i0)
    assert empty == (i0 - i0) == (i2 - i2) == (i3 - i3)

    assert (i0 | i4 - i4) == i0
    assert (i4 | i0 - i0) == i4

    assert tuple(empty) == ()
    assert len(tuple(i0)) == 7
    assert len(tuple(i4)) == 70


def test_other():
    empty = IntCrowd()
    i0 = IntCrowd(Interval[1, 2:4, 2:4, 7, 2:4])
    assert repr(i0) == 'IntCrowd(Interval[1, 2:4, 2:4, 7, 2:4])'
    assert len(tuple(i0)) == 8
    assert empty != i0 == (i0 | i0) == (i0 | i0 | i0)
    assert i0 - i0 == empty
    assert i0 | empty == i0
    assert empty | i0 == i0
    assert empty | empty == empty
    assert empty - empty == empty
    assert empty - i0 == empty
    assert i0 - i0 | i0 == i0


def test_addition_bug():
    intervals = (Interval[1, 1], Interval[0:2, 0])
    int_crowd = IntCrowd(intervals)
    assert set(itertools.chain.from_iterable(intervals)) == set(int_crowd)
