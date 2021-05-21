# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from grid_royale import utils
from grid_royale import gamey
from grid_royale.gamey.sample_games import blackjack

def test_gamey():
    assert 'g' in repr(gamey)


def test_blackjack_demo():
    with utils.OutputCapturer() as output_capturer:
        blackjack.demo(n_training_phases=100)

    assert "let's run the old comparison again" in output_capturer.output
