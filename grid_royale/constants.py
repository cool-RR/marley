# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import pathlib
import logging
import string as string_module

SHOT_REWARD = -10
COLLISION_REWARD = -5
STARTING_REWARD = 0
NOTHING_REWARD = -1
FOOD_REWARD = 10

VISION_RANGE = 4
VISION_SIZE = VISION_RANGE * 2 + 1

LETTERS = string_module.ascii_uppercase

DEFAULT_BOARD_SIZE = 6 # 12
DEFAULT_N_FOOD_TILES = 5 # 20
DEFAULT_N_PLAYERS = 3 # 10

grid_royale_folder = pathlib.Path.home() / '.grid_royale'
config_path = grid_royale_folder / 'config.json'



