# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import pathlib
import logging.config
import datetime as datetime_module
import re
import shlex

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



def setup(*, verbose: bool = False, log_to_file: bool = True) -> None:
    if log_to_file:
        constants.logs_folder.mkdir(parents=True, exist_ok=True)
        clean_logs_folder()
        now = datetime_module.datetime.now()
        log_file_stem = re.sub('[^0-9]+', '-', now.isoformat(timespec='milliseconds'))
        assert re.fullmatch('[0-9-]+', log_file_stem)
        log_file_path: pathlib.Path = constants.logs_folder / f'{log_file_stem}.log'
        assert not log_file_path.exists()

        file_logging_setup = {
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_path,
                'mode': 'w',
                'formatter': 'verbose',
            },
        }

    else:
        file_logging_setup = {}


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
    if log_to_file:
        logger.info(f'Log file: {shlex.quote(str(log_file_path))}')

