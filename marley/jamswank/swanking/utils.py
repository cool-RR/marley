# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
import datetime as datetime_module
from typing import Iterable, Mapping, Union, Type, Optional, Any
import itertools
import collections.abc
import re
import functools
import types
import numpy as np

from ..jamming.utils import type_to_name, name_to_type



def camel_case_to_lower_case(s: str) -> str:
    '''
    Convert a string from camel-case to lower-case.

    Example:

        camel_case_to_lower_case('HelloWorld') == 'hello_world'

    '''
    return re.sub(r'(((?<=[a-z])[A-Z])|([A-Z](?![A-Z]|$)))', r'_\1', s).lower().strip('_')

