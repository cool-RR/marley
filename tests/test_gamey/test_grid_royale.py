# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import grid_royale
from grid_royale import State, Observation, Action, Culture
from grid_royale import gamey

def test_simple():
    culture = Culture(n_players=1, board_size=4)
    state_0 = culture.make_initial_state(n_food_tiles=2)
    assert len(state_0.food_positions) == 2
    (letter,) = culture.player_id_to_strategy
    assert isinstance(state_0, State)
    assert state_0.board_size == 4
    assert not state_0.is_end
    assert not state_0.bullets
    state_1 = state_0.get_next_state_from_actions({letter: Action.shoot_right})
    state_2 = state_1.get_next_state_from_actions({letter: Action.shoot_left})
    assert state_1.bullets or state_2.bullets


