# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import pathlib
import logging
import functools
import json

from marley.jamswank.jamming.jam_file_database import JamFileDatabase
from .marley_swank_database import MarleySwankDatabase

@functools.cache
def read_config() -> dict:
    try:
        content = config_path.read_text()
    except FileNotFoundError:
        return {}
    return json.loads(content)



marley_folder: pathlib.Path = pathlib.Path.home() / '.marley'
config_path: pathlib.Path = marley_folder / 'config.json'
logs_folder: pathlib.Path = marley_folder / 'logs'
config = read_config() # todo: Shouldn't be on import

if 'jam_database_path' in config:
    jam_database_folder = pathlib.Path(config['jam_database_path'])
    assert jam_database_folder.exists()
else:
    jam_database_folder: pathlib.Path = marley_folder / 'jam_database'


jam_file_database = JamFileDatabase(jam_database_folder)
swank_database = MarleySwankDatabase(jam_file_database)

