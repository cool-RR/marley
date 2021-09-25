# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import os
import logging
import collections

# Avoid TensorFlow spam:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.getLogger('tensorflow').addFilter(
    lambda record: 'Tracing is expensive and the excessive' not in record.msg
)

from . import pycompat
from . import utils
from . import constants
from . import sharknado # todo remove
# from . import gamey
# from . import commanding

__VersionInfo = collections.namedtuple('VersionInfo',
                                       ('major', 'minor', 'micro'))

__version__ = '0.0.4'
__version_info__ = __VersionInfo(*(map(int, __version__.split('.'))))

def import_all_worlds():
    import marley.worlds.grid_royale
    import marley.worlds.blackjack


del os, logging, collections, __VersionInfo # Avoid polluting the namespace
