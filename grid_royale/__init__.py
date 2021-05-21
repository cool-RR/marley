# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import os
import collections
import logging

# Avoid TensorFlow spam:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').addFilter(
    lambda record: 'Tracing is expensive and the excessive' not in record.msg
)

from .core import *
from .commands import grid_royale

__VersionInfo = collections.namedtuple('VersionInfo',
                                       ('major', 'minor', 'micro'))

__version__ = '0.3.0'
__version_info__ = __VersionInfo(*(map(int, __version__.split('.'))))

del os, collections, __VersionInfo # Avoid polluting the namespace
