# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''
This module defines the GridRoyale game and all its rules.

# GridRoyale game rules #

Here are the rules of the game in simple form. For all the gritty details, check out the
`get_next_state_from_actions` method.

The world is a 2-dimensional grid, typically 20x20 in size. Each player is on a cell in the grid.
There can't be more than one player in a cell.

Each player basically wants to eat as much food as possible, avoid getting shot, and avoid colliding
with other players, walls and the edge of the board.

On each turn, each player does an action. It can be either moving, shooting or building a wall
(a.k.a "walling").

**Moving**: Moving is essential because it lets you walk towards food and away from bullets.

When a player is moving, it sometimes collides with a player, a wall or the edge of the
board. It collides with another player if that other player is trying to move into the same cell, or
is in the same cell and not moving. If there's a collision, the player's move is unsuccessful. It
remains in its old position, and it takes a punishment, currently -5 points.

If the move is successful, *and* the new cell has food, the player eats the food and gains 10
points. (Assuming it didn't get shot, more about that later.) The food will be removed from the
game, and a new piece of food will be spawned in a random free space on the board.

**Shooting**: A player may shoot a bullet in any of the four directions. The bullet will continue
travelling until it either hits a player, giving it a punishment of -10, or disappears off the edge
of the board, or hits a wall and destroys it. Shooting is useful to scare off other creatures who
might otherwise eat your food-- Assuming they're intelligent enough to avoid bullets.

Multiple bullets can freely pass through each other on their trajectory. They don't hit each other.

**Walling**: A player may build a wall in any of the four directions. A wall is a barrier that
blocks any player from entering that cell. The only way to remove a wall is to shoot it. Walls are
useful to prevent another player from moving into a certain area, especially if multiple walls are
built next to each other.
'''


from __future__ import annotations

import functools
import json
import pathlib
import string as string_module
import statistics
import webbrowser
import time
import itertools
import random
import logging
import numbers
import io
import collections
from typing import (Optional, Tuple, Union, Container, Hashable, Iterator, Mapping,
                    Iterable, Any, Dict, FrozenSet, Callable, Type, Sequence, Set)
import re
import dataclasses
import datetime as datetime_module
import click

import numpy as np
import scipy.special
import more_itertools
import keras

from . import gamey
from . import utils
from .utils import zip
from .gamey.utils import ImmutableDict
from .vectoring import Vector, Step, Position, Translation, Vicinity


SHOT_REWARD = -10
COLLISION_REWARD = -5
STARTING_REWARD = 0
NOTHING_REWARD = -1
FOOD_REWARD = 10

VISION_RANGE = 4
VISION_SIZE = VISION_RANGE * 2 + 1

LETTERS = string_module.ascii_uppercase

DEFAULT_BOARD_SIZE = 12
DEFAULT_CONCURRENT_FOOD_TILES = 20
DEFAULT_N_PLAYERS = 10

grid_royale_folder = pathlib.Path.home() / '.grid_royale'
config_path = grid_royale_folder / 'config.json'

logging.getLogger('tensorflow').addFilter(
    lambda record: 'Tracing is expensive and the excessive' not in record.msg
)

_action_neuron_cache = {}

@functools.cache
def read_config() -> dict:
    try:
        content = config_path.read_text()
    except FileNotFoundError:
        return {}
    return json.loads(content)

@functools.cache
def get_games_folder() -> pathlib.Path:
    config = read_config()
    try:
        games_folder_string = config['games_folder']
    except KeyError:
        return grid_royale_folder / 'games'
    games_folder = pathlib.Path(games_folder_string)
    assert games_folder.is_dir()
    return games_folder


@dataclasses.dataclass(order=True, frozen=True)
class Action(gamey.Action):
    move: Optional[Step]
    shoot: Optional[Step]
    wall: Optional[Step]

    def __post_init__(self):
        if tuple(self).count(None) != 2:
            raise ValueError

    __iter__ = lambda self: iter((self.move, self.shoot, self.wall))


    def _to_neural(self) -> np.ndarray:
        result = np.zeros(12, dtype=np.float64)
        result[Action.all_actions.index(self)] = 1
        return result

    def to_neural(self) -> np.ndarray:
        return _action_neuron_cache[self]

    @property
    def name(self) -> str:
        if self.move is not None:
            return self.move.name
        elif self.shoot is not None:
            return f'shoot_{self.shoot.name}'
        else:
            return f'wall_{self.wall.name}'



(Action.up, Action.right, Action.down, Action.left) = \
    Action.all_move_actions = (Action(Step.up, None, None), Action(Step.right, None, None),
                                   Action(Step.down, None, None), Action(Step.left, None, None))

(Action.shoot_up, Action.shoot_right, Action.shoot_down, Action.shoot_left) = \
    Action.all_shoot_actions = (Action(None, Step.up, None), Action(None, Step.right, None),
                                   Action(None, Step.down, None), Action(None, Step.left, None))

(Action.wall_up, Action.wall_right, Action.wall_down, Action.wall_left) = \
    Action.all_wall_actions = (Action(None, None, Step.up), Action(None, None, Step.right),
                                   Action(None, None, Step.down), Action(None, None, Step.left))

Action.all_actions = Action.all_move_actions + Action.all_shoot_actions + Action.all_wall_actions

for action in Action:
    _action_neuron_cache[action] = action._to_neural()


@dataclasses.dataclass(order=True, frozen=True)
class Bullet:
    '''A bullet that's shot by one player and might hit another player.'''
    position: Position
    direction: Step

    def get_next_bullet(self):
        return Bullet(position=(self.position + self.direction), direction=self.direction)

    def get_previous_bullet(self):
        return Bullet(position=(self.position - self.direction), direction=self.direction)




class _BaseGrid:
    '''Base class that represents a 2-dimensional square grid.'''
    board_size: int
    def __contains__(self, position: Position) -> bool:
        return 0 <= min(position) <= max(position) <= self.board_size - 1

    @staticmethod
    def iterate_random_positions(board_size: int) -> Iterator[Position]:
        while True:
            yield Position(
                random.randint(0, board_size - 1),
                random.randint(0, board_size - 1),
            )



class Observation(_BaseGrid, gamey.Observation):

    is_end = False
    legal_move_actions = Action.all_move_actions

    def __init__(self, state: Optional[State], position: Position, *,
                 letter: str, score: int, last_action: Optional[Action],
                 reward: int = STARTING_REWARD) -> None:
        assert len(letter) == 1
        self.state = state
        self.position = position
        self.letter = letter.upper()
        self.score = score
        self.reward = reward
        self.last_action = last_action


    @property
    def board_size(self) -> int:
        return self.state.board_size

    @property
    def culture(self) -> Culture:
        return self.state.culture

    @property
    def legal_actions(self):
        return (
            *Action.all_move_actions,
            *(Action.all_shoot_actions if self.state.allow_shooting else ()),
            *(Action.all_wall_actions if self.state.allow_walling else ()),
        )


    @functools.cache
    def to_neural(self) -> np.ndarray:

        ### Calculating grid subarray: #############################################################
        #                                                                                          #
        grid_array = np.zeros((self.board_size, self.board_size, 10))
        relative_player_position = Position(self.board_size // 2, self.board_size // 2)
        translation = relative_player_position - self.position

        for relative_position in Position.iterate_all(self.board_size):
            absolute_position: Position = relative_position - translation
            if absolute_position in self.state.food_positions:
                grid_array[tuple(relative_position) + (1,)] = 1
            if absolute_position not in self:
                grid_array[tuple(relative_position) + (0,)] = 1
            elif (bullets := self.state.bullets.get(absolute_position, None)):
                for bullet in bullets:
                    grid_array[tuple(relative_position) + (2 + bullet.direction.index,)] = 1
            elif absolute_position in self.state.living_wall_positions:
                grid_array[tuple(relative_position) + (6,)] = 1
            elif absolute_position in self.state.destroyed_wall_positions:
                grid_array[tuple(relative_position) + (7,)] = 1
            elif (observation := self.state.position_to_observation.get(absolute_position, None)):
                observation: Observation
                if observation.letter == self.letter:
                    grid_array[tuple(relative_position) + (8,)] = 1
                else:
                    grid_array[tuple(relative_position) + (9,)] = 1
        #                                                                                          #
        ### Finished calculating grid subarray. ####################################################

        ### Calculating sequential subarray: #######################################################
        #                                                                                          #
        last_action_neurons = (self.last_action.to_neural() if self.last_action is not None
                               else np.zeros(len(Action)))
        sequential_array = np.concatenate((
            last_action_neurons,
        ))
        #                                                                                          #
        ### Finished calculating sequential subarray. ##############################################

        array = np.zeros((1,), dtype=[('grid', int, grid_array.shape),
                                      ('sequential', np.float64, sequential_array.shape)])
        array[0]['grid'] = grid_array
        array[0]['sequential'] = sequential_array
        return array


    @functools.cache
    def processed_distances(self, vicinity: Vicinity) -> numbers.Real:
        field_of_view = self.position.field_of_view(vicinity, self.board_size)

        for distance_to_food, positions in enumerate(field_of_view, start=1):
            if positions & self.state.food_positions:
                break
        else:
            distance_to_food = float('inf')

        for distance_to_living_wall, positions in enumerate(field_of_view, start=1):
            if positions & self.state.living_wall_positions:
                break
        else:
            distance_to_living_wall = float('inf')


        distances_to_other_players = []

        for i, positions in enumerate(field_of_view, start=1):
            if any((player_position in positions) for player_position in
                   self.state.position_to_observation):
                distances_to_other_players.append(i)
                break
        else:
            distances_to_other_players.append(float('inf'))


        for strategy in self.culture.strategies:
            for i, positions in enumerate(field_of_view, start=1):
                strategies = (
                    self.culture.player_id_to_strategy[observation.letter]
                    for position in positions if (observation :=
                                 self.state.position_to_observation.get(position, None)) is not None
                )
                if strategy in strategies:
                    distances_to_other_players.append(i)
                    break
            else:
                distances_to_other_players.append(float('inf'))

        assert len(distances_to_other_players) == len(self.culture.strategies) + 1

        distances_to_bullets = []
        for direction in Step.all_steps:
            for i, positions in enumerate(field_of_view, start=1):
                bullets = itertools.chain.from_iterable(
                    self.state.bullets.get(position, ()) for position in positions
                )
                if any(bullet.direction == direction for bullet in bullets):
                    distances_to_bullets.append(i)
                    break
            else:
                distances_to_bullets.append(float('inf'))


        return tuple(self._distance_base ** (-distance)
                     for distance in ([distance_to_food, distance_to_living_wall] +
                                      distances_to_other_players + distances_to_bullets))


    @functools.cache
    def processed_distance_to_edge(self, vicinity: Vicinity) -> numbers.Real:
        position = self.position
        for i in itertools.count():
            if position not in self:
                distance_to_edge = i
                break
            position += vicinity
        return 1.2 ** (-distance_to_edge)

    @property
    def cute_score(self) -> int:
        return self.score - min((self.position @ food_position
                                for food_position in self.state.food_positions),
                                default=(-100))

    def p(self) -> None:
        print(self.state.ascii)


class State(_BaseGrid, gamey.State):

    Observation = Observation
    Action = Action
    is_end = False

    def __init__(self, culture: Culture, *, board_size: int,
                 player_id_to_observation: ImmutableDict[str, Observation],
                 food_positions: FrozenSet[Position], allow_shooting: bool = True,
                 allow_walling: bool = True,
                 bullets: ImmutableDict[Position, FrozenSet[Bullet]] = ImmutableDict(),
                 living_wall_positions: FrozenSet[Position],
                 destroyed_wall_positions: FrozenSet[Position]) -> None:
        self.culture = culture
        self.player_id_to_observation = player_id_to_observation
        self.bullets = bullets
        self.allow_shooting = allow_shooting
        self.allow_walling = allow_walling
        assert all(bullets.values()) # No empty sets in this bad boy.
        self.all_bullets = frozenset(itertools.chain.from_iterable(bullets.values()))
        self.food_positions = food_positions
        self.board_size = board_size
        self.position_to_observation = ImmutableDict(
            {observation.position: observation for observation in player_id_to_observation.values()}
        )
        self.living_wall_positions = living_wall_positions
        self.destroyed_wall_positions = destroyed_wall_positions
        self.wall_positions = living_wall_positions | destroyed_wall_positions

    def _reduce(self) -> tuple:
        return (
            type(self),
            frozenset(
                (letter, observation.position, observation.score, observation.reward,
                 observation.last_action) for letter, observation in
                self.player_id_to_observation.items()
            ),
            self.bullets, self.food_positions, self.board_size, self.living_wall_positions,
            self.destroyed_wall_positions
        )

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, State) and self._reduce() == other._reduce()

    def __hash__(self) -> int:
        return hash(self._reduce())


    @staticmethod
    def make_initial(culture: Culture, *, board_size: int, starting_score: int = 0,
                     allow_shooting: bool = True, allow_walling: bool = True,
                     concurrent_food_tiles: int = DEFAULT_CONCURRENT_FOOD_TILES) -> State:

        n_players = len(culture.strategies)
        random_positions_firehose = utils.iterate_deduplicated(
                                     State.iterate_random_positions(board_size=board_size))
        random_positions = tuple(
            more_itertools.islice_extended(
                random_positions_firehose)[:(n_players + concurrent_food_tiles)]
        )

        player_positions = random_positions[:n_players]
        food_positions = frozenset(random_positions[n_players:])
        assert len(food_positions) == concurrent_food_tiles

        player_id_to_observation = {}
        for letter, player_position in zip(LETTERS, player_positions):
            player_id_to_observation[letter] = Observation(state=None, position=player_position,
                                                        score=starting_score, letter=letter,
                                                        last_action=None)

        state = State(
            culture=culture,
            board_size=board_size,
            player_id_to_observation=ImmutableDict(player_id_to_observation),
            food_positions=food_positions,
            allow_shooting=allow_shooting,
            allow_walling=allow_walling,
            living_wall_positions=frozenset(),
            destroyed_wall_positions=frozenset(),
        )

        for observation in player_id_to_observation.values():
            observation.state = state

        return state


    def get_next_state_from_actions(self, player_id_to_action: Mapping[Position, Action]) -> State:
        new_player_position_to_olds = collections.defaultdict(set)
        wip_living_wall_positions = set(self.living_wall_positions)
        wip_destroyed_wall_positions = set() # Ignoring old destroyed walls, they're gone.

        for letter, action in player_id_to_action.items():
            action: Action
            old_observation = self.player_id_to_observation[letter]
            assert action in old_observation.legal_actions
            old_player_position = old_observation.position
            if action.move:
                new_player_position_to_olds[old_player_position +
                                                               action.move].add(old_player_position)
            else:
                new_player_position_to_olds[old_player_position].add(old_player_position)
                if action.wall and (wall_position := old_player_position + action.wall) in self:
                    wip_living_wall_positions.add(wall_position)


        ############################################################################################
        ### Figuring out which players collided: ###################################################
        #                                                                                          #
        # There are four types of collisions:
        # 1. A player trying to move out of the board.
        # 2. A player trying to move to a wall, living or destroyed.
        # 3. Two or more players that try to move into the same position.
        # 4. Two players that are trying to move into each other's positions.
        # 5. Any players that are trying to move into the old position of a player that had one
        #    of the two collisions above, and is therefore still occupying that position.

        collided_player_positions = set()


        while True:
            for new_player_position, old_player_positions in new_player_position_to_olds.items():
                if (new_player_position not in self) or \
                                                 (new_player_position in wip_living_wall_positions):
                    # This is either a type 1 or a type 2 collision.
                    collided_player_positions |= old_player_positions
                    del new_player_position_to_olds[new_player_position]
                    for old_player_position in old_player_positions:
                        new_player_position_to_olds[old_player_position].add(old_player_position)
                        if old_player_position in wip_living_wall_positions:
                            wip_living_wall_positions.remove(old_player_position)

                    # We modified the dict while iterating, let's restart the loop:
                    break

                if len(old_player_positions) >= 2:
                    # This is either a type 3 or a type 5 collision.
                    collided_player_positions |= old_player_positions
                    del new_player_position_to_olds[new_player_position]
                    for old_player_position in old_player_positions:
                        new_player_position_to_olds[old_player_position].add(old_player_position)

                    # We modified the dict while iterating, let's restart the loop:
                    break

                if (len(old_player_positions) == 1 and
                    ((old_player_position := more_itertools.one(old_player_positions)) !=
                      new_player_position) and new_player_position_to_olds.get(
                                               old_player_position, None) == {new_player_position}):
                    # Type 4 collision.
                    collided_player_positions |= {old_player_position, new_player_position}
                    new_player_position_to_olds[new_player_position] = {new_player_position}
                    new_player_position_to_olds[old_player_position] = {old_player_position}

                    # We modified the dict while iterating, let's restart the loop:
                    break

            else:
                # We already found all collisions, if any.
                break
        #                                                                                          #
        ### Finished figuring out which players collided. ##########################################
        ############################################################################################

        new_player_position_to_old = {
            new_player_position: more_itertools.one(old_player_positions) for
            new_player_position, old_player_positions in new_player_position_to_olds.items()
        }
        del new_player_position_to_olds # Prevent confusion

        # Disallowing building walls where there is a player that didn't move:
        for living_wall_position in tuple(wip_living_wall_positions):
            try:
                old_player_position = new_player_position_to_old[living_wall_position]
            except KeyError:
                pass
            else:
                assert old_player_position == living_wall_position
                wip_living_wall_positions.remove(living_wall_position)

        ############################################################################################
        ### Figuring out bullets: ##################################################################
        #                                                                                          #

        # Todo: This section needs a lot of tests!

        wip_bullets: Mapping[Position, Set[Bullet]] = collections.defaultdict(set)

        # Continuing trajectory for existing bullets:
        for bullet in self.all_bullets:
            new_bullet = bullet.get_next_bullet()
            wip_bullets[new_bullet.position].add(new_bullet)

        # Processing new bullets that were just fired:
        for letter, action in player_id_to_action.items():
            if action.shoot is not None:
                player_position = self.player_id_to_observation[letter].position
                new_bullet = Bullet(player_position + action.shoot, action.shoot)
                wip_bullets[new_bullet.position].add(new_bullet)

        # Clearing bullets out of board:
        for position in [position for position, bullets in wip_bullets.items()
                         if (position not in self)]:
            del wip_bullets[position]

        # Figuring out which walls were shot, removing these bullets:
        for living_wall_position in tuple(wip_living_wall_positions):
            if wip_bullets.pop(living_wall_position, None):
                wip_living_wall_positions.remove(living_wall_position)
                wip_destroyed_wall_positions.add(living_wall_position)


        # Figuring out which players were shot, removing these bullets:
        new_player_positions_that_were_shot = set()
        for new_player_position, old_player_position in new_player_position_to_old.items():
            if wip_bullets.get(new_player_position, None):
                # Common shooting case.
                del wip_bullets[new_player_position]
                new_player_positions_that_were_shot.add(new_player_position)
            elif translation := new_player_position - old_player_position: # Player moved
                oncoming_bullet_direction = - Step(*translation)
                oncoming_bullets = {bullet for bullet in wip_bullets.get(old_player_position, ()) if
                                    bullet.direction == oncoming_bullet_direction}
                if oncoming_bullets:
                    # Less-common shooting case: The player walked towards an oncoming bullet,
                    # switching places with it.
                    (oncoming_bullet,) = oncoming_bullets
                    wip_bullets[old_player_position].remove(oncoming_bullet)
                    new_player_positions_that_were_shot.add(new_player_position)


        bullets = ImmutableDict({
            position: frozenset(bullets) for position, bullets in wip_bullets.items() if bullets
        })

        #                                                                                          #
        ### Finished figuring out bullets. #########################################################
        ############################################################################################

        ############################################################################################
        ### Figuring out food: #####################################################################
        #                                                                                          #
        new_player_positions_that_ate_food = set()
        wip_food_positions = set(self.food_positions)
        random_positions_firehose = utils.iterate_deduplicated(
            State.iterate_random_positions(board_size=self.board_size),
            seen=(set(itertools.chain(new_player_position_to_old, self.position_to_observation)) |
                  self.food_positions)
        )
        for new_player_position in new_player_position_to_old:
            if new_player_position in self.food_positions:
                wip_food_positions.remove(new_player_position)
                wip_food_positions.add(next(random_positions_firehose))
                new_player_positions_that_ate_food.add(new_player_position)

        assert len(wip_food_positions) == len(self.food_positions)

        #                                                                                          #
        ### Finished figuring out food. ############################################################
        ############################################################################################

        player_id_to_observation = {}

        for new_player_position, old_player_position in new_player_position_to_old.items():
            letter = self.position_to_observation[old_player_position].letter
            old_observation: Observation = self.player_id_to_observation[letter]

            reward = (
                SHOT_REWARD if new_player_position in new_player_positions_that_were_shot else
                COLLISION_REWARD if new_player_position in collided_player_positions else
                FOOD_REWARD if new_player_position in new_player_positions_that_ate_food else
                NOTHING_REWARD
            )

            player_id_to_observation[letter] = Observation(
                state=None,
                position=new_player_position,
                score=old_observation.score + reward,
                reward=reward,
                letter=letter,
                last_action=player_id_to_action[letter]
            )


        state = State(
            culture=self.culture, board_size=self.board_size,
            player_id_to_observation=ImmutableDict(player_id_to_observation),
            food_positions=frozenset(wip_food_positions), bullets=bullets,
            allow_shooting=self.allow_shooting, allow_walling=self.allow_walling,
            living_wall_positions=frozenset(wip_living_wall_positions),
            destroyed_wall_positions=frozenset(wip_destroyed_wall_positions),
        )

        for observation in player_id_to_observation.values():
            observation.state = state

        return state

    @property
    def ascii(self) -> None:
        '''A nice ascii representation of all the objects in the state.'''
        string_io = io.StringIO()
        for position in Position.iterate_all(self):
            if position.x == 0 and position.y != 0:
                string_io.write('|\n')
            if position in self.position_to_observation:
                observation = self.position_to_observation[position]
                letter = (observation.letter.lower() if observation.reward < NOTHING_REWARD
                          else observation.letter)
                string_io.write(letter)
            elif (bullets := self.bullets.get(position, None)):
                if len(bullets) >= 2:
                    string_io.write('Ӿ')
                else:
                    string_io.write(more_itertools.one(bullets).direction.ascii)


            elif position in self.food_positions: # But no bullets
                string_io.write('·')
            else:
                string_io.write(' ')

        string_io.write('|\n')
        string_io.write('‾' * self.board_size)
        string_io.write('\n')

        return string_io.getvalue()

    def p(self) -> None:
        print(self.ascii)


    def write_to_folder(self, folder: pathlib.Path, *, chunk: int = 10,
                        max_length: Optional[int] = None) -> Iterator[State]:
        from .animating import animate

        ### Preparing folder: #################################################
        #                                                                     #
        if not folder.exists():
            folder.mkdir(parents=True)
        assert folder.is_dir()
        assert not any(folder.iterdir())
        #                                                                     #
        ### Finished preparing folder. ########################################

        paths = ((folder / f'{i:06d}.json') for i in range(10 ** 6))
        state_iterator, state_iterator_copy = itertools.tee(
                                             self.culture.iterate_game(self, max_length=max_length))
        transition_iterator = animate(state_iterator_copy)
        for path in paths:
            chunked_iterator = more_itertools.islice_extended(
                                      zip(state_iterator, transition_iterator, strict=True))[:chunk]
            transition_chunk = []
            for state, transition in chunked_iterator:
                transition_chunk.append(transition)
                yield state
            if not transition_chunk:
                # Game ended, either naturally or reached `max_length`.
                return
            assert len(transition_chunk) <= chunk
            with path.open('w') as file:
                json.dump(transition_chunk, file)

    def write_to_game_folder(self, *, chunk: int = 10,
                             max_length: Optional[int] = None) -> Iterator[State]:
        now = datetime_module.datetime.now()
        game_folder_name = now.isoformat(sep='-', timespec='microseconds'
                                                               ).replace(':', '-').replace('.', '-')
        game_folder = get_games_folder() / game_folder_name
        # print(f'Writing to {game_folder.name} ...')
        for state in self.write_to_folder(game_folder, chunk=chunk, max_length=max_length):
            yield state
        # print(f'Finished writing to {game_folder.name} .')


    @property
    def average_reward(self) -> numbers.Real:
        return statistics.mean(observation.reward for observation in
                               self.player_id_to_observation.values())





class Culture(gamey.ModelFreeLearningCulture):

    def __init__(self, n_players: int = DEFAULT_N_PLAYERS, *, board_size: int = DEFAULT_BOARD_SIZE,
                 allow_shooting: bool = True, allow_walling: bool = True,
                 concurrent_food_tiles: int = DEFAULT_CONCURRENT_FOOD_TILES,
                 strategies: Optional[Sequence[_GridRoyaleStrategy]] = None) -> None:
        if strategies is not None:
            assert n_players == len(strategies)
        self.board_size = board_size
        self.allow_shooting = allow_shooting
        self.allow_walling = allow_walling
        self.default_concurrent_food_tiles = concurrent_food_tiles
        self.strategies = tuple(strategies or (Strategy(self) for _ in range(n_players)))
        # self.executor = concurrent.futures.ProcessPoolExecutor(5)
        gamey.Culture.__init__(self, state_type=State,
                               player_id_to_strategy=dict(zip(LETTERS, self.strategies)))


    def make_initial_state(self, *, concurrent_food_tiles: Optional[int] = None) -> State:
        concurrent_food_tiles = (concurrent_food_tiles if concurrent_food_tiles is not None
                                 else self.default_concurrent_food_tiles)
        return State.make_initial(
            self, board_size=self.board_size, allow_shooting=self.allow_shooting,
            allow_walling=self.allow_walling, concurrent_food_tiles=concurrent_food_tiles
        )



class _GridRoyaleStrategy(gamey.Strategy):
    State = State


class SimpleStrategy(_GridRoyaleStrategy):

    def __init__(self, epsilon: int = 0.2) -> None:
        self.epsilon = epsilon


    def decide_action_for_observation(self, observation: Observation) -> Action:
        if random.random() <= self.epsilon or not observation.state.food_positions:
            return random.choice(observation.legal_actions)
        else:
            closest_food_position = min(
                (food_position for food_position in observation.state.food_positions),
                key=lambda food_position: observation.position @ food_position
            )
            desired_translation = closest_food_position - observation.position
            dimension = random.choice(
                tuple(dimension for dimension, delta in enumerate(desired_translation) if delta)
            )
            return (Action(np.sign(desired_translation.x), 0) if dimension == 0
                    else Action(0, np.sign(desired_translation.y)))



class Strategy(_GridRoyaleStrategy, gamey.ModelFreeLearningStrategy):

    def __init__(self, culture: Culture, **kwargs) -> None:
        self.culture = culture
        gamey.ModelFreeLearningStrategy.__init__(self, training_batch_size=10, **kwargs)

    def create_model(self, observation: Observation, action: Action) -> keras.Model:
        observation_neural = observation.to_neural()

        grid_input = keras.Input(
            shape=observation_neural[0]['grid'].shape,
            name='grid_input'
        )
        grid_0 = keras.layers.Conv2D(
            16, 2, activation='relu',
            kernel_initializer='orthogonal'
        )(grid_input)
        _ = keras.layers.Dropout(rate=0.1)(grid_0)
        _ = keras.layers.Conv2D(
            16, 2, activation='relu',
            kernel_initializer='orthogonal'
        )(_)
        _ = keras.layers.Dropout(rate=0.1)(_)
        _ = keras.layers.Conv2D(
            16, 2, activation='relu',
            kernel_initializer='orthogonal'
        )(_)
        _ = keras.layers.Flatten()(_)
        grid_output = keras.layers.Dropout(rate=0.1)(_)

        sequential_input = keras.Input(
            shape=observation_neural[0]['sequential'].shape,
            name='sequential_input'
        )

        concatenate_layer = keras.layers.concatenate([grid_output, sequential_input])

        _ = keras.layers.Dense(
            128, activation='relu',
        )(concatenate_layer)
        _ = keras.layers.Dropout(rate=0.1)(_)
        _ = keras.layers.Dense(
            128, activation='relu',
        )(_)
        _ = keras.layers.Dropout(rate=0.1)(_)
        output = keras.layers.Dense(
            more_itertools.one(action.to_neural().shape),
        )(_)
        model = keras.Model([grid_input, sequential_input], output)
        model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
        return model





@click.group()
def grid_royale() -> None:
    pass

from . import server

@grid_royale.command()
@click.option('--board-size', type=int, default=DEFAULT_BOARD_SIZE)
@click.option('--n-players', type=int, default=DEFAULT_N_PLAYERS)
@click.option('--concurrent-food-tiles', type=int, default=DEFAULT_CONCURRENT_FOOD_TILES)
@click.option('--allow-shooting/--no-shooting', default=True)
@click.option('--allow-walling/--no-walling', default=False)
@click.option('--pre-train/--dont-pre-train', default=False)
@click.option('--browser/--no-browser', 'open_browser', default=True)
@click.option('--host', default=server.DEFAULT_HOST)
@click.option('--port', default=server.DEFAULT_PORT)
@click.option('--max-length', default=None, type=int)
def play(*, board_size: int, n_players: int, concurrent_food_tiles: int, allow_shooting: bool,
         allow_walling: bool, pre_train: bool, open_browser: bool, host: str, port: str,
         max_length: Optional[int] = None) -> None:
    with server.ServerThread(host=host, port=port, quiet=True) as server_thread:
        culture = Culture(n_players=n_players, board_size=board_size,
                          concurrent_food_tiles=concurrent_food_tiles,
                          allow_shooting=allow_shooting, allow_walling=allow_walling)
        state = culture.make_initial_state()

        if open_browser:
            click.echo(f'Opening {server_thread.url} in your browser to view the game.')
            webbrowser.open_new(server_thread.url)
        else:
            click.echo(f'Open {server_thread.url} in your browser to view the game.')

        if pre_train:
            pre_train_n_games = 4
            pre_train_max_length = 20
            click.echo(f'Pre-training {pre_train_n_games} games, each with '
                       f'{pre_train_max_length} states...', nl=False)
            for _ in culture.multi_game_train(n_total_games=pre_train_n_games,
                                              max_length=pre_train_max_length):
                click.echo('.', nl=False)
            click.echo(' Done pre-training.')

        if max_length is None:
            click.echo(f'Calculating states in the simulation, press Ctrl-C to stop.')
        else:
            click.echo(f'Calculating {max_length} states, press Ctrl-C to stop.')

        for state in state.write_to_game_folder(max_length=max_length):
            pass
        click.echo(f'Finished calculating {max_length} states, still serving forever.')
        while True:
            time.sleep(0.1)



@grid_royale.command()
@click.option('--host', default=server.DEFAULT_HOST)
@click.option('--port', default=server.DEFAULT_PORT)
def serve(*, host: str, port: str) -> None:
    with server.ServerThread(host=host, port=port) as server_thread:
        click.echo(f'Open {server_thread.url} in your browser to view the game.')
        while True:
            time.sleep(0.1)

@grid_royale.command()
@click.argument('game_name', default='blackjack')
@click.option('--n-training-games', default=1_000)
@click.option('--n-evaluation-games', default=100)
def sample(game_name: str, n_training_games: int, n_evaluation_games: int):
    assert re.match('^[a-z_][a-z0-9_]{1,100}', game_name)
    from grid_royale.gamey.sample_games import blackjack, griddler
    games = {
        'blackjack': blackjack,
        'griddler': griddler,
    }
    game = games[game_name]
    game.demo(n_training_games=n_training_games,
              n_evaluation_games=n_evaluation_games)

@grid_royale.command()
@click.option('--n-training-games', default=1_000)
@click.option('--n-evaluation-games', default=100)
def griddler(n_training_games: int, n_evaluation_games: int):
    from grid_royale.gamey.sample_games import griddler
    griddler.demo(n_training_games=n_training_games,
                  n_evaluation_games=n_evaluation_games)


if __name__ == '__main__':
    grid_royale()
