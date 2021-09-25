# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import itertools

from .pointing import Point
from .. import int_crowding


def intervals_to_points(intervals: Iterable[int_crowding.Interval]) -> Tuple[Point, ...]:
    return tuple(sorted(set(itertools.chain.from_iterable(intervals))))

def points_to_intervals(points: Iterable[Point]) -> Tuple[int_crowding.Interval, ...]:
    current_intervals = tuple(sorted(map(Point.as_interval, points)))
    if not current_intervals:
        return ()
    dimensions = current_intervals[0].dimensions
    def fluff():
        start_i = current_start.as_tuple()[i]
        end_i = current_current.as_tuple()[i]
        money_item = (start_i, end_i + 1) if start_i != end_i else start_i
        return int_crowding.Interval(current_current.as_tuple()[:i] + (money_item,) +
                                     current_current.as_tuple()[dimensions - d:])

    for d in range(dimensions):
        i = dimensions - d - 1
        new_intervals = []
        for _prefix, intervals in itertools.groupby(current_intervals,
                                                    lambda interval: interval.as_tuple()[:i]):
            current_start = current_current = next(intervals)

            for interval in intervals:
                if ((interval.as_tuple()[i] == current_current.as_tuple()[i] + 1) and
                    (interval.as_tuple()[i + 1:] == current_current.as_tuple()[i + 1:])):
                    current_current = interval
                else:
                    new_intervals.append(fluff())
                    current_start = current_current = interval
            new_intervals.append(fluff())

        current_intervals = tuple(new_intervals)

    return current_intervals

def canonicalize_intervals(intervals: Iterable[int_crowding.Interval] = ()) -> \
                                                                  Tuple[int_crowding.Interval, ...]:
    return points_to_intervals(intervals_to_points(intervals))


