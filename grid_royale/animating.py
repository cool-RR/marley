# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''This module defines the `animate` function that turns GridRoyale states into animation data.'''


from __future__ import annotations

from typing import (Optional, Tuple, Union, Container, Hashable, Iterator,
                    Iterable, Any, Dict, FrozenSet, Callable, Type, Sequence)
import itertools
import math

from .vectoring import Step
from .base import (State, Observation, SHOT_REWARD, COLLISION_REWARD, STARTING_REWARD,
                   NOTHING_REWARD, FOOD_REWARD, Bullet)

from . import gamey


_directions = ((0, -1), (1, 0), (0, 1), (-1, 0))

color_map = {
    SHOT_REWARD: (0x99, 0x44, 0x44),
    COLLISION_REWARD: (0x88, 0x88, 0x22),
    STARTING_REWARD: (0x55, 0x55, 0x55),
    NOTHING_REWARD: (0x55, 0x55, 0x55),
    FOOD_REWARD: (0x55, 0x55, 0x55),
}

def animate(states: Iterable[State]) -> Iterator[dict]:
    state_iterator = iter(states)
    try:
        first_state = next(state_iterator)
    except StopIteration:
        return ()

    double_state_iterator = gamey.utils.iterate_windowed_pairs(
        itertools.chain((first_state, first_state), state_iterator)
    )

    for old_state, new_state in double_state_iterator:

        old_state: State
        new_state: State

        ############################################################################################
        ### Processing players: ####################################################################
        #                                                                                          #
        players = []

        for letter, old_observation in old_state.items():
            old_observation: Observation
            new_observation: Observation = new_state[letter]
            players.append((
                letter, (tuple(old_observation.position), tuple(new_observation.position)),
                (color_map[old_observation.reward], color_map[new_observation.reward]), (1, 1),
            ))
        #                                                                                          #
        ### Finished processing players. ###########################################################
        ############################################################################################

        ############################################################################################
        ### Processing food: #######################################################################
        #                                                                                          #
        food = []
        old_food_positions = set(map(tuple, old_state.food_positions))
        new_food_positions = set(map(tuple, new_state.food_positions))

        created_food_positions = new_food_positions - old_food_positions
        deleted_food_positions = old_food_positions - new_food_positions
        remaining_food_positions = old_food_positions & new_food_positions

        for created_food_position in created_food_positions:
            food.append((created_food_position, (0, 1)))

        for deleted_food_position in deleted_food_positions:
            food.append((deleted_food_position, (1, 0)))

        for remaining_food_position in remaining_food_positions:
            food.append((remaining_food_position, (1, 1)))

        #                                                                                          #
        ### Finished processing food. ##############################################################
        ############################################################################################

        ############################################################################################
        ### Processing bullets: ####################################################################
        #                                                                                          #
        bullets = []
        old_bullets_wip = set(itertools.chain.from_iterable(old_state.bullets.values()))
        new_bullets_wip = set(itertools.chain.from_iterable(new_state.bullets.values()))

        while old_bullets_wip:
            old_bullet: Bullet = old_bullets_wip.pop()
            desired_new_bullet = old_bullet.get_next_bullet()
            try:
                new_bullets_wip.remove(desired_new_bullet)
            except KeyError:
                bullet_hit = True
            else:
                bullet_hit = False

            bullets.append((
                (tuple(old_bullet.position), tuple(desired_new_bullet.position)),
                old_bullet.direction.angle_from_top,
                (1, 1 - bullet_hit)
            ))

        while new_bullets_wip:
            new_bullet: Bullet = new_bullets_wip.pop()
            bullets.append((
                (tuple(new_bullet.get_previous_bullet().position), tuple(new_bullet.position)),
                new_bullet.direction.angle_from_top,
                (0, 1)
            ))

        #                                                                                          #
        ### Finished processing bullets. ###########################################################
        ############################################################################################

        ############################################################################################
        ### Processing walls: ######################################################################
        #                                                                                          #
        walls_dict = {}

        for wall_position in old_state.living_wall_positions:
            walls_dict.setdefault(wall_position, [0, 0])[0] = 1

        for wall_position in old_state.destroyed_wall_positions:
            walls_dict.setdefault(wall_position, [0, 0])[0] = 0.5

        for wall_position in new_state.living_wall_positions:
            walls_dict.setdefault(wall_position, [0, 0])[1] = 1

        for wall_position in new_state.destroyed_wall_positions:
            walls_dict.setdefault(wall_position, [0, 0])[1] = 0.5

        walls = tuple(sorted((tuple(wall_position), tuple(opacities)) for wall_position, opacities
                             in walls_dict.items()))

        #                                                                                          #
        ### Finished processing walls. #############################################################
        ############################################################################################

        yield {
            'board_size': old_state.board_size,
            'players': tuple(sorted(players)),
            'food': tuple(sorted(food)),
            'bullets': tuple(sorted(bullets)),
            'walls': walls,
        }
