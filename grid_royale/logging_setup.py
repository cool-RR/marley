# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import pathlib
import logging.config
import datetime as datetime_module
import re
import shlex

from . import constants


MAX_LOG_FILES_TO_KEEP = 100


def setup():
    constants.logs_folder.mkdir(parents=True, exist_ok=True)
    logs = tuple(constants.logs_folder.iterdir())
    if len(logs) >= MAX_LOG_FILES_TO_KEEP - 1:
        logs_to_delete = sorted(logs, key=lambda path: path.stat().st_ctime)[
                                                                     : -(MAX_LOG_FILES_TO_KEEP - 1)]
        for log_to_delete in logs_to_delete:
            log_to_delete.unlink()

    now = datetime_module.datetime.now()
    log_stem = re.sub('[^0-9]+', '-', now.isoformat(timespec='milliseconds'))
    assert re.fullmatch('[0-9-]+', log_stem)
    log: pathlib.Path = constants.logs_folder / f'{log_stem}.log'
    assert not log.exists()

    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'verbose': {
                    'format': '{levelname} {asctime} {module} {process:d} {thread:d} | {message}',
                    'style': '{',
                },
                'simple': {
                    'format': '{message}',
                    'style': '{',
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'simple',
                },
                'file': {
                    'level': 'DEBUG',
                    'class': 'logging.FileHandler',
                    'filename': log,
                    'mode': 'w',
                    'formatter': 'verbose',
                },
            },
            'root': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
            },
        }
    )

    logger = logging.getLogger(__name__)
    logger.info(f'Log file: {shlex.quote(str(log))}')