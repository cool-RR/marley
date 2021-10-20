# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import pathlib
import logging.config
import datetime as datetime_module
import re
import shlex
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)


from . import constants


MAX_LOG_FILES_TO_KEEP = 100

class NoisyThirdPartyFilter(logging.Filter):
    def filter(self, record):
        return record.name != 'filelock'

def clean_logs_folder() -> None:
    logs = tuple(constants.logs_folder.iterdir())
    if len(logs) >= MAX_LOG_FILES_TO_KEEP - 1:
        logs_to_delete = sorted(logs, key=lambda path: path.stat().st_ctime)[
                                                                 : -(MAX_LOG_FILES_TO_KEEP - 1)]
        for log_to_delete in logs_to_delete:
            log_to_delete.unlink()

log_file_path: Optional[pathlib.Path] = None
did_logging_setup: bool = False
is_verbose: Optional[bool] = None

def setup(*, verbose: bool = False, log_to_file: bool = True,
          existing_log_file_path: Optional[pathlib.Path] = None) -> None:
    global log_file_path, did_logging_setup, is_verbose
    if did_logging_setup:
        return
    is_verbose = verbose
    if log_to_file:
        if existing_log_file_path is not None:
            assert log_file_path is None
            log_file_path = existing_log_file_path
        else: # existing_log_file_path is None
            constants.logs_folder.mkdir(parents=True, exist_ok=True)
            clean_logs_folder()
            now = datetime_module.datetime.now()
            log_file_stem = re.sub('[^0-9]+', '-', now.isoformat(timespec='milliseconds'))
            assert re.fullmatch('[0-9-]+', log_file_stem)
            log_file_path = constants.logs_folder / f'{log_file_stem}.log'
            assert not log_file_path.exists()

        file_logging_setup = {
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_path,
                'mode': 'a',
                'formatter': 'verbose',
            },
        }

    else:
        assert existing_log_file_path is None
        assert log_file_path is None
        file_logging_setup = {}


    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'verbose': {
                    'format': '{levelname} {asctime} {module} p{process:d} t{thread:d} | {message}',
                    'style': '{',
                },
                'simple': {
                    'format': '{message}',
                    'style': '{',
                },
            },
            'filters': {
                'noisy_third_party_filter': {
                    '()': NoisyThirdPartyFilter,
                },
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG' if verbose else 'INFO',
                    'formatter': 'simple',
                    'filters': ('noisy_third_party_filter',),
                },
                **file_logging_setup,
            },
            'root': {
                'handlers': ('console', *(('file',) if log_to_file else ())),
                'level': 'DEBUG',
            },
        }
    )

    logger = logging.getLogger(__name__)
    if log_to_file and not existing_log_file_path:
        logger.info(f'Log file: {shlex.quote(str(log_file_path))}')
    did_logging_setup = True


def get_logging_kwargs() -> dict:
    assert did_logging_setup
    return {
        'verbose': is_verbose,
        'log_to_file': (log_file_path is not None),
        'existing_log_file_path': log_file_path
    }

