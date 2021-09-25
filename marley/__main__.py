# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''Entry point for starting GridRoyale.'''

import logging
import contextlib

import click

logger = logging.getLogger(__name__)

def is_exit_0_exception(exception: Exception) -> bool:
    return (
        (isinstance(exception, SystemExit) and exception.code == 0) or
        (isinstance(exception, click.exceptions.Exit) and exception.exit_code == 0)
    )


@contextlib.contextmanager
def run_and_log_exception():
    try:
        yield
    except BaseException as base_exception:
        if not is_exit_0_exception(base_exception):
            logger.exception('Marley exited because of an exception.')
            raise SystemExit(1) from base_exception


if __name__ == '__main__':
    import marley.commanding
    # Import all the worlds to register their command groups:
    marley.import_all_worlds()

    with run_and_log_exception():
        marley.commanding.marley()
