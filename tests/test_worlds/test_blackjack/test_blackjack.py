# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import more_itertools
import pytest

from marley.worlds import blackjack

@pytest.mark.skip()
def test_blackjack_demo():
    blackjack.demo(n_training_phases=3, n_evaluation_games=20)
