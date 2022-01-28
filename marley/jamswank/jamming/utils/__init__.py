# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from typing import Iterable, Sequence, Iterable
import itertools
import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional, Any
import itertools
import collections.abc
import re
import functools
import types
import numpy as np


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


_address_pattern = re.compile(
    r"^(?P<address>([a-zA-Z_][0-9a-zA-Z_]*)(\.[a-zA-Z_][0-9a-zA-Z_]*)*)$"
)
'''Pattern for Python addresses, like 'email.encoders'.'''



def _get_parent_and_dict_from_namespace(namespace):
    '''
    Extract the parent object and `dict` from `namespace`.

    For the `namespace`, the user can give either a parent object
    (`getattr(namespace, address) is obj`) or a `dict`-like namespace
    (`namespace[address] is obj`).

    Returns `(parent_object, namespace_dict)`.
    '''

    if hasattr(namespace, '__getitem__') and hasattr(namespace, 'keys'):
        parent_object = None
        namespace_dict = namespace

    else:
        parent_object = namespace
        namespace_dict = vars(parent_object)

    return (parent_object, namespace_dict)


def is_address(string):
    return bool(_address_pattern.match(string))


def import_if_exists(module_name):
    '''
    Import module by name and return it, only if it exists.
    '''
    try:
        return __import__(module_name)
    except ModuleNotFoundError:
        return None

def cute_import(module_name):
    '''
    Import a module.

    This function has several advantages over `__import__`:

     1. It avoids the weird `fromlist=['']` that you need to give `__import__`
        in order for it to return the specific module you requested instead of
        the outermost package, and

     2. It avoids a weird bug in Linux, where importing using `__import__` can
        lead to a `module.__name__` containing two consecutive dots.

    '''
    if '.' in module_name:
        package_name, submodule_name = module_name.rsplit('.', 1)
        package = __import__(module_name)
        return functools.reduce(getattr,
                                [package] + module_name.split('.')[1:])
    else:
        return __import__(module_name)




def get_object_by_address(address, root=None, namespace={}):
    r'''
    Get an object by its address.

    For example:

        >>> get_object_by_address('email.encoders')
        <module 'email.encoders' from 'c:\Python27\lib\email\encoders.pyc'>

    `root` is an object (usually a module) whose attributes will be looked at
    when searching for the object. `namespace` is a `dict` whose keys will be
    searched as well.
    '''
    # todo: should know what exception this will raise if the address is bad /
    # object doesn't exist.

    assert is_address(address)

    if not _address_pattern.match(address):
        raise ValueError("'%s' is not a legal address." % address)

    ###########################################################################
    # Before we start, we do some pre-processing of `root` and `namespace`:   #

    # We are letting the user input (base)strings for `root` and `namespace`,
    # so if he did that, we'll get the actual objects.

    if root:
        # First for `root`:
        if isinstance(root, str):
            root = get_object_by_address(root)
        root_short_name = root.__name__.rsplit('.', 1)[-1]

    if namespace not in (None, {}):
        # And then for `namespace`:
        if isinstance(namespace, str):
            namespace = get_object_by_address(namespace)

        parent_object, namespace_dict = _get_parent_and_dict_from_namespace(namespace)
    else:
        parent_object, namespace_dict = None, None


    # Finished pre-processing `root` and `namespace`.                         #
    ###########################################################################


    ###########################################################################
    # The implementation is recursive: We handle the case of a single-level
    # address, like 'email'. If we get a multi-level address (i.e. contains a
    # dot,) like 'email.encoders', we use this function twice, first to get
    # `email`, and then from it to get `email.encoders`.

    if '.' not in address:

        ### Here we solve the basic case of a single-level address: ###########
        #                                                                     #

        # Let's rule out the easy option that the requested object is the root:
        if root and (address == root_short_name):
            return root

        if parent_object is not None:

            if isinstance(parent_object, types.ModuleType) and hasattr(parent_object, '__path__'):

                # `parent_object` is a package. The wanted object may be a
                # module. Let's try importing it:

                import_if_exists('.'.join((parent_object.__name__, address)),)
                # Not keeping reference, just importing so we could get later.

        # We know we have a `namespace_dict` to take the object from, and we
        # might have a `parent_object` we can take the object from by using
        # `getattr`. We always have a `namespace_dict`, but not always a
        # `parent_object`.
        #


        # We are going to prefer to do `getattr` from `parent_object`, if one
        # exists, rather than using `namespace_dict`. This is because some
        # attributes may not be present on an object's `__dict__`, and we want
        # to be able to catch them:

        # The first place we'll try to take the object from is the
        # `parent_object`. We try this before `namespace_dict` because
        # `parent_object` may have `__getattr__` or similar magic and our
        # object might be found through that:
        if (parent_object is not None) and hasattr(parent_object, address):
            return getattr(parent_object, address)

        # Next is the `namespace_dict`:
        elif namespace_dict and (address in namespace_dict):
            return namespace_dict[address]

        # Last two options:
        else:
            try:
                # It may be a built-in:
                return eval(address)
            except Exception:
                # Or a module:
                return cute_import(address)

        #                                                                     #
        ### Finished solving the basic case of a single-level address. ########


    else: # '.' in address

        ### If we get a composite address, we solve recursively: ##############
        #                                                                     #

        first_object_address, second_object_address = address.rsplit('.', 1)

        first_object = get_object_by_address(first_object_address, root=root, namespace=namespace)

        second_object = get_object_by_address(second_object_address, namespace=first_object)

        return second_object

        #                                                                     #
        ### Finished solving recursively for a composite address. #############


@functools.cache
def type_to_name(type_: Type, *, safety_check: bool = True) -> str:
    assert isinstance(type_, type)

    # Special handling for Numpy's overly clever _DTypeMeta metaclass:
    if type_.__name__.startswith('dtype') and (type(type_) is type(np.dtype)):
        return 'numpy.dtype'

    name = (type_.__name__ if type_.__module__ == 'builtins'
            else f'{type_.__module__}.{type_.__name__}')

    if safety_check:
        assert name_to_type(name, safety_check=False) is type_
    return name


@functools.cache
def name_to_type(name: str, *, safety_check: bool = True) -> Type:
    type_ = get_object_by_address(name)
    if safety_check:
        assert type_to_name(type_, safety_check=False) == name
    return type_

