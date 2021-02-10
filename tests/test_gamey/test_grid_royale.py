# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import more_itertools

import grid_royale
from grid_royale import State, Observation, Action, Culture, Game, Policy
from grid_royale import gamey

def test_simple():

    culture = Culture({'A': Policy()})
    game = Game.from_state_culture(State.make_initial(n_players=1), culture)
    assert len(game) == 1
    (state_0,) = game.states
    assert isinstance(state_0, State)
    assert len(state_0.food_positions) == grid_royale.base.DEFAULT_N_FOOD_TILES
    assert len(state_0) == 1 # One player makes for one observation.
    assert state_0.board_size == grid_royale.base.DEFAULT_BOARD_SIZE
    assert not state_0.is_end
    assert not state_0.bullets
    state_0_again, state_1, state_2 = more_itertools.islice_extended(game)[:3]
    assert state_0 is state_0_again
    assert type(state_0) is type(state_1) is type(state_2) is State
    assert len(game) == 3
    observation: Observation = state_2['A']
    assert observation.legal_actions



