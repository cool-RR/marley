# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from grid_royale import utils
from grid_royale import gamey
from grid_royale.gamey.sample_games import blackjack

def test_gamey():
    assert 'g' in repr(gamey)


def test_blackjack_demo():
    blackjack.demo(n_training_phases=3, n_evaluation_games=20)
