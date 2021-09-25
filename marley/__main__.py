# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''Entry point for starting GridRoyale.'''

import logging
import contextlib

import click

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    import marley.commanding
    marley.commanding.marley()