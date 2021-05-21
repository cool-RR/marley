# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import click
import webbrowser
import pathlib
import re
import time
from typing import (Optional, Tuple, Union, Container, Hashable, Iterator, Mapping,
                    Iterable, Any, Dict, FrozenSet, Callable, Type, Sequence, Set)

from .constants import LETTERS, DEFAULT_BOARD_SIZE, DEFAULT_N_FOOD_TILES, DEFAULT_N_PLAYERS
from .core import State, Culture, Policy, NaiveCulture, NaivePolicy, Game
from . import utils
from . import gamey


@click.group()
def grid_royale() -> None:
    pass

from . import server

@grid_royale.command()
@click.option('--board-size', type=int, default=DEFAULT_BOARD_SIZE)
@click.option('--n-players', type=int, default=DEFAULT_N_PLAYERS)
@click.option('--n-food-tiles', type=int, default=DEFAULT_N_FOOD_TILES)
@click.option('--allow-shooting/--no-shooting', default=True)
@click.option('--allow-walling/--no-walling', default=False)
@click.option('--pre-train/--dont-pre-train', default=True)
@click.option('--pre-train-n-games', type=int, default=20)
@click.option('--pre-train-n-phases', type=int, default=10)
@click.option('--browser/--no-browser', 'open_browser', default=True)
@click.option('--host', default=server.DEFAULT_HOST)
@click.option('--port', default=server.DEFAULT_PORT)
@click.option('--max-length', default=None, type=int)
def play(*, board_size: int, n_players: int, n_food_tiles: int, allow_shooting: bool,
         allow_walling: bool, pre_train: bool, pre_train_n_games: int, pre_train_n_phases: int,
         open_browser: bool, host: str, port: str, max_length: Optional[int] = None) -> None:
    with server.ServerThread(host=host, port=port, quiet=True) as server_thread:

        if open_browser:
            click.echo(f'Opening {server_thread.url} in your browser to view the game.')
            webbrowser.open_new(server_thread.url)
        else:
            click.echo(f'Open {server_thread.url} in your browser to view the game.')

        make_initial_state = lambda: (
            State.make_initial(
                n_players=n_players, board_size=board_size, allow_shooting=allow_shooting,
                allow_walling=allow_walling, n_food_tiles=n_food_tiles
            )
        )

        letters = LETTERS[:n_players]

        culture = Culture({letter: Policy(board_size=board_size)
                           for letter in letters[:n_players]})

        if pre_train:
            with utils.NiceTaskShower('Pre-training') as nice_task_shower:
                for culture in culture.train_iterate(make_initial_state, n_games=pre_train_n_games,
                                                     max_game_length=30,
                                                     n_phases=pre_train_n_phases):
                    nice_task_shower.dot()
                # The last `culture` from the for loop is used below.

        game = Game.from_state_culture(make_initial_state(), culture)

        if max_length is None:
            click.echo(f'Calculating states in the simulation, press Ctrl-C to stop.')
        else:
            click.echo(f'Calculating {max_length} states, press Ctrl-C to stop.')

        for state in game.write_to_game_folder(max_length=max_length):
            pass
        click.echo(f'Finished calculating {max_length} states, still serving forever.')
        while True:
            time.sleep(0.1)


@grid_royale.command()
@click.option('--n-players', type=int, default=3)
@click.option('--n-food-tiles', type=int, default=5)
@click.option('--allow-shooting/--no-shooting', default=True)
@click.option('--allow-walling/--no-walling', default=False)
@click.option('--browser/--no-browser', 'open_browser', default=True)
@click.option('--host', default=server.DEFAULT_HOST)
@click.option('--port', default=server.DEFAULT_PORT)
@click.option('--max-length', default=300, type=int)
def demo(*, n_players: int, n_food_tiles: int, allow_shooting: bool, allow_walling: bool,
         open_browser: bool, host: str, port: str, max_length: Optional[int] = None) -> None:
    board_size = 6
    with server.ServerThread(host=host, port=port, quiet=True) as server_thread:

        if open_browser:
            click.echo(f'Opening {server_thread.url} in your browser to view the game.')
            webbrowser.open_new(server_thread.url)
        else:
            click.echo(f'Open {server_thread.url} in your browser to view the game.')

        make_initial_state = lambda: (
            State.make_initial(
                n_players=n_players, board_size=board_size, allow_shooting=allow_shooting,
                allow_walling=allow_walling, n_food_tiles=n_food_tiles
            )
        )

        letters = LETTERS[:n_players]

        serialized_model = (pathlib.Path(__file__).parent / 'demo.h5').read_bytes()
        culture = Culture(
            {letter: Policy(board_size=board_size, serialized_models=(serialized_model,),
                            n_models=1, training_period=float('inf'))
             for letter in letters[:n_players]}
        )

        game = Game.from_state_culture(make_initial_state(), culture)

        if max_length is None:
            click.echo(f'Calculating states in the simulation, press Ctrl-C to stop.')
        else:
            click.echo(f'Calculating {max_length} states, press Ctrl-C to stop.')

        for state in game.write_to_game_folder(max_length=max_length):
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
@click.option('--n-training-states', default=10_000)
@click.option('--n-evaluation-games', default=100)
def sample(game_name: str, n_training_states: int, n_evaluation_games: int):
    assert re.match('^[a-z_][a-z0-9_]{1,100}', game_name)
    from grid_royale.gamey.sample_games import blackjack
    games = {
        'blackjack': blackjack,
    }
    game = games[game_name]
    game.demo(n_training_states=n_training_states,
              n_evaluation_games=n_evaluation_games)

