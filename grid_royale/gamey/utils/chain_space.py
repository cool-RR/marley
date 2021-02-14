# Copyright 2020 Ram Rachum.
# This program is distributed under the MIT license.

from __future__ import generator_stop
from __future__ import annotations

import collections.abc
import functools
import threading
import itertools

from . import iterating

infinity = float('inf')

class _SENTINEL:
    '''Sentinel used to detect the end of an iterable.'''


def _convert_index_to_exhaustion_point(index):
    '''
    Convert an index to an "exhaustion point".

    The index may be either an integer or infinity.

    "Exhaustion point" means "until which index do we need to exhaust the
    internal iterator." If an index of `3` was requested, we need to exhaust it
    to index `3`, but if `-7` was requested, we have no choice but to exhaust
    the iterator completely (i.e. to `infinity`, actually the last element,)
    because only then we could know which member is the seventh-to-last.
    '''
    assert isinstance(index, int) or index == infinity
    if index >= 0:
        return index
    else: # i < 0
        return infinity


def _with_lock(method):
    '''Decorator for using the `LazyTuple`'s lock.'''
    @functools.wraps(method)
    def inner(self, *args, **kwargs):
        with self.lock:
            return method(*args, **kwargs)
    return inner


@functools.total_ordering
class LazyTuple(collections.abc.Sequence):
    '''
    A lazy tuple which requests as few values as possible from its iterator.

    Wrap your iterators with `LazyTuple` and enjoy tuple-ish features like
    indexed access, comparisons, length measuring, element counting and more.

    Example:

        def my_generator():
            yield from ('hello', 'world', 'have', 'fun')

        lazy_tuple = LazyTuple(my_generator())

        assert lazy_tuple[2] == 'have'
        assert len(lazy_tuple) == 4

    `LazyTuple` holds the given iterable and pulls items out of it. It pulls as
    few items as it possibly can. For example, if you ask for the third
    element, it will pull exactly three elements and then return the third one.

    Some actions require exhausting the entire iterator. For example, checking
    the `LazyTuple` length, or doing indexex access with a negative index.
    (e.g. asking for the seventh-to-last element.)

    If you're passing in an iterator you definitely know to be infinite,
    specify `definitely_infinite=True`.
    '''

    def __init__(self, iterable, *, definitely_infinite=False):
        was_given_a_sequence = isinstance(iterable, collections.abc.Sequence) \
                                        and not isinstance(iterable, LazyTuple)

        self.is_exhausted = True if was_given_a_sequence else False
        '''Flag saying whether the internal iterator is tobag exhausted.'''

        self.collected_data = iterable if was_given_a_sequence else []
        '''All the items that were collected from the iterable.'''

        self._iterator = None if was_given_a_sequence else iter(iterable)
        '''The internal iterator from which we get data.'''

        self.definitely_infinite = definitely_infinite
        '''
        The iterator is definitely infinite.

        The iterator might still be infinite if this is `False`, but if it's
        `True` then it's definitely infinite.
        '''

        self.lock = threading.Lock()
        '''Lock used while exhausting to make `LazyTuple` thread-safe.'''


    @classmethod
    def factory(cls, *, definitely_infinite=False):
        '''
        Decorator to make generators return a `LazyTuple`.

        Example:

            @LazyTuple.factory()
            def my_generator():
                yield from ['hello', 'world', 'have', 'fun']

        This works on any function that returns an iterator. todo: Make it work
        on iterator classes.
        '''
        def decorator(function):
            @functools.wraps(function)
            def inner(*args, **kwargs):
                return cls(function(*args, **kwargs),
                           definitely_infinite=definitely_infinite)
            return inner
        return decorator


    @property
    def known_length(self):
        '''
        The number of items which have been taken from the internal iterator.
        '''
        return len(self.collected_data)


    def exhaust(self, i=infinity):
        '''
        Take items from the internal iterators and save them.

        This will take enough items so we will have `i` items in total,
        including the items we had before.
        '''
        if self.is_exhausted:
            return

        elif isinstance(i, int) or i == infinity:
            exhaustion_point = _convert_index_to_exhaustion_point(i)

        else:
            assert isinstance(i, slice)
            raise NotImplementedError

            # # todo: can be smart and figure out if it's an empty slice and then
            # # not exhaust.

            # canonical_slice = sequence_tools.CanonicalSlice(i)

            # exhaustion_point = max(
                # _convert_index_to_exhaustion_point(canonical_slice.start),
                # _convert_index_to_exhaustion_point(canonical_slice.stop)
            # )

            # if canonical_slice.step > 0: # Compensating for excluded last item:
                # exhaustion_point -= 1

        while len(self.collected_data) <= exhaustion_point:
            try:
                with self.lock:
                    self.collected_data.append(next(self._iterator))
            except StopIteration:
                self.is_exhausted = True
                break


    def __getitem__(self, i):
        '''Get item by index, either an integer index or a slice.'''
        self.exhaust(i)
        result = self.collected_data[i]
        if isinstance(i, slice):
            return tuple(result)
        else:
            return result


    def __len__(self):
        if self.definitely_infinite:
            return 0 # Unfortunately infinity isn't supported.
        else:
            self.exhaust()
            return len(self.collected_data)


    def __eq__(self, other):
        if not isinstance(other, LazyTuple):
            return False
        for i, j in itertools.zip_longest(self, other, fillvalue=_SENTINEL):
            if (i is _SENTINEL) or (j is _SENTINEL):
                return False
            if i != j:
                return False
        return True


    def __ne__(self, other):
        return not self.__eq__(other)


    def __bool__(self):
        try: next(iter(self))
        except StopIteration: return False
        else: return True


    def __lt__(self, other):
        if not self and other:
            return True
        elif self and not other:
            return False
        elif not self and not other:
            return False
        for a, b in itertools.zip_longest(self, other,
                                           fillvalue=_SENTINEL):
            if a is _SENTINEL:
                # `self` ran out. Now there can be two cases: (a) `other` ran
                # out too or (b) `other` didn't run out yet. In case of (a), we
                # have `self == other`, and in case of (b), we have `self <
                # other`. In any case, `self <= other is True` so we can
                # unconditionally return `True`.
                return True
            elif b is _SENTINEL:
                assert a is not _SENTINEL
                return False
            elif a == b:
                continue
            elif a < b:
                return True
            else:
                assert a > b
                return False


    def __repr__(self):
        '''
        Return a human-readeable representation of the `LazyTuple`.

        Example:

            <LazyTuple: (1, 2, 3, ...)>

        The '...' denotes a non-exhausted lazy tuple.
        '''
        if self.is_exhausted:
            inner = repr(self.collected_data)

        else: # not self.exhausted
            if self.collected_data == []:
                inner = '(...)'
            else:
                inner = '%s...' % repr(self.collected_data)
        return '<%s: %s>' % (self.__class__.__name__, inner)


    def __add__(self, other):
        return tuple(self) + tuple(other)


    def __radd__(self, other):
        return tuple(other) + tuple(self)


    def __mul__(self, other):
        return tuple(self).__mul__(other)


    def __rmul__(self, other):
        return tuple(self).__rmul__(other)


    def __hash__(self):
        '''
        Get the `LazyTuple`'s hash.

        Note: Hashing the `LazyTuple` will completely exhaust it.
        '''
        if self.definitely_infinite:
            raise TypeError("An infinite `LazyTuple` isn't hashable.")
        else:
            self.exhaust()
            return hash(tuple(self))



class ChainSpace(collections.abc.Sequence):
    '''
    A space of sequences chained together.

    This is similar to `itertools.chain`, except that items can be fetched by
    index number rather than just iteration.

    Example:

        >>> chain_space = ChainSpace(('abc', (1, 2, 3)))
        >>> chain_space
        <ChainSpace: 3+3>
        >>> chain_space[4]
        2
        >>> tuple(chain_space)
        ('a', 'b', 'c', 1, 2, 3)
        >>> chain_space.index(2)
        4

    '''
    def __init__(self, sequences):
        self.sequences = LazyTuple(sequences)
        self.accumulated_lengths = LazyTuple(
            itertools.accumulate(map(len, self.sequences), initial=0)
        )

    def __len__(self):
        return self.accumulated_lengths[-1]

    def __repr__(self):
        return '<%s: %s>' % (
            type(self).__name__,
            '+'.join(str(len(sequence)) for sequence in self.sequences),
        )

    def __getitem__(self, i):
        if isinstance(i, slice):
            raise NotImplementedError
        assert isinstance(i, int)
        if i <= -1:
            fixed_i += len(self) # Exhausting the chain space.
        else:
            fixed_i = i
        if fixed_i < 0:
            raise IndexError(i)
        required_length = fixed_i + 1
        if self.accumulated_lengths.is_exhausted and required_length > len(self):
            raise IndexError

        iterator = zip(self.sequences, iterating.iterate_windowed_pairs(self.accumulated_lengths))
        for sequence, (previous_accumulated_length, accumulated_length) in iterator:
            if accumulated_length >= required_length:
                assert previous_accumulated_length < required_length
                return sequence[i - previous_accumulated_length]
        else:
            raise IndexError(i)

    def __iter__(self):
        return itertools.chain.from_iterable(self.sequences)

    _reduced = property(lambda self: (type(self), self.sequences))

    __eq__ = lambda self, other: (isinstance(other, ChainSpace) and
                                  self._reduced == other._reduced)

    def __contains__(self, item):
        return any(item in sequence for sequence in self.sequences
                   if (not isinstance(sequence, str) or isinstance(item, str)))

    def index(self, item):
        '''Get the index number of `item` in this space.'''
        for sequence, accumulated_length in zip(self.sequences,
                                                self.accumulated_lengths):
            try:
                index_in_sequence = sequence.index(item)
            except ValueError:
                pass
            except TypeError:
                assert isinstance(sequence, (str, bytes)) and \
                                           (not isinstance(item, (str, bytes)))
            else:
                return index_in_sequence + accumulated_length
        else:
            raise ValueError

    def __bool__(self):
        try: next(iter(self))
        except StopIteration: return False
        else: return True

