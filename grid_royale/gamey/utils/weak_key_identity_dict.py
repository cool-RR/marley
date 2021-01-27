# Copyright 2009-2017 Ram Rachum.
# This program is distributed under the MIT license.

'''
Defines the `WeakKeyIdentityDict` class.

See its documentation for more details.
'''
# todo: revamp

from __future__ import generator_stop

import weakref
import collections


__all__ = ['WeakKeyIdentityDict']


class IdentityRef(weakref.ref):
    '''A weak reference to an object, hashed by identity and not contents.'''

    def __init__(self, thing, callback=None):
        weakref.ref.__init__(self, thing, callback)
        self._hash = id(thing)


    def __hash__(self):
        return self._hash


class WeakKeyIdentityDict(collections.abc.MutableMapping):
    """
    A weak key dictionary which cares about the keys' identities.

    This is a fork of `weakref.WeakKeyDictionary`. Like in the original
    `WeakKeyDictionary`, the keys are referenced weakly, so if there are no
    more references to the key, it gets removed from this dict.

    The difference is that `WeakKeyIdentityDict` cares about the keys'
    identities and not their contents, so even unhashable objects like lists
    can be used as keys. The value will be tied to the object's identity and
    not its contents.
    """

    def __init__(self, dict_=None):
        self.data = {}
        def remove(k, selfref=weakref.ref(self)):
            self = selfref()
            if self is not None:
                del self.data[k]
        self._remove = remove
        if dict_ is not None: self.update(dict_)


    def __delitem__(self, key):
        del self.data[IdentityRef(key)]


    def __getitem__(self, key):
        return self.data[IdentityRef(key)]


    def __repr__(self):
        return f"<WeakKeyIdentityDict at {id(self)}>"


    def __setitem__(self, key, value):
        self.data[IdentityRef(key, self._remove)] = value


    def copy(self):
        """ D.copy() -> a shallow copy of D """
        new = WeakKeyIdentityDict()
        for key, value in self.data.items():
            o = key()
            if o is not None:
                new[o] = value
        return new


    def get(self, key, default=None):
        """ D.get(k[,d]) -> D[k] if k in D, else d.  d defaults to None. """
        return self.data.get(IdentityRef(key),default)


    def __contains__(self, key):
        try:
            wr = IdentityRef(key)
        except TypeError:
            return 0
        return wr in self.data


    has_key = __contains__


    def items(self):
        """ D.items() -> list of D's (key, value) pairs, as 2-tuples """
        L = []
        for key, value in list(self.data.items()):
            o = key()
            if o is not None:
                L.append((o, value))
        return L


    def iteritems(self):
        """ D.iteritems() -> an iterator over the (key, value) items of D """
        for wr, value in self.data.items():
            key = wr()
            if key is not None:
                yield key, value


    def iterkeyrefs(self):
        """Return an iterator that yields the weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return iter(self.data.keys())


    def iterkeys(self):
        """ D.iterkeys() -> an iterator over the keys of D """
        for wr in self.data.keys():
            obj = wr()
            if obj is not None:
                yield obj

    def __iter__(self):
        return iter(self.keys())


    def itervalues(self):
        """ D.itervalues() -> an iterator over the values of D """
        return iter(self.data.values())


    def keyrefs(self):
        """Return a list of weak references to the keys.

        The references are not guaranteed to be 'live' at the time
        they are used, so the result of calling the references needs
        to be checked before being used.  This can be used to avoid
        creating references that will cause the garbage collector to
        keep the keys around longer than needed.

        """
        return list(self.data.keys())


    def keys(self):
        """ D.keys() -> list of D's keys """
        L = []
        for wr in list(self.data.keys()):
            o = wr()
            if o is not None:
                L.append(o)
        return L


    def popitem(self):
        """ D.popitem() -> (k, v), remove and return some (key, value) pair
        as a 2-tuple; but raise KeyError if D is empty """
        while True:
            key, value = self.data.popitem()
            o = key()
            if o is not None:
                return o, value


    def pop(self, key, *args):
        """ D.pop(k[,d]) -> v, remove specified key and return the
        corresponding value. If key is not found, d is returned if given,
        otherwise KeyError is raised """
        return self.data.pop(IdentityRef(key), *args)


    def setdefault(self, key, default=None):
        """D.setdefault(k[,d]) -> D.get(k,d), also set D[k]=d if k not in D"""
        return self.data.setdefault(IdentityRef(key, self._remove),default)


    def update(self, dict=None, **kwargs):
        """ D.update(E, **F) -> None. Update D from E and F: for k in E: D[k] =
        E[k] (if E has keys else: for (k, v) in E: D[k] = v) then: for k in F:
        D[k] = F[k] """

        d = self.data
        if dict is not None:
            if not hasattr(dict, "items"):
                dict = type({})(dict)
            for key, value in dict.items():
                d[IdentityRef(key, self._remove)] = value
        if len(kwargs):
            self.update(kwargs)


    def __len__(self):
        return len(self.data)

