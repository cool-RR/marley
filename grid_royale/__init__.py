# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import os
import collections

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Avoid TensorFlow spam

from .base import *

__VersionInfo = collections.namedtuple('VersionInfo',
                                       ('major', 'minor', 'micro'))

__version__ = '0.0.8'
__version_info__ = __VersionInfo(*(map(int, __version__.split('.'))))

del os, collections, __VersionInfo # Avoid polluting the namespace
