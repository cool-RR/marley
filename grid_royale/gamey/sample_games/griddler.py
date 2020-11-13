# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations
from typing import Iterable, Tuple, Any

import sys
import itertools
import random
import enum
import functools
import numpy as np

import more_itertools

from grid_royale import gamey


class GriddlerAction(gamey.ActionEnum):
    down = 'down'
    right = 'right'

GriddlerAction.all_actions = (GriddlerAction.down, GriddlerAction.right)


class GriddlerState(gamey.SinglePlayerState):

    Action = GriddlerAction

    reward: int

    def __init__(self, x: int = 0, y: int = 0) -> None:
        self.x = x
        self.y = y
        if x == y == 2:
            self.reward = 1
            self.is_end = True
        elif {x, y} <= set(range(3)):
            self.reward = 0
            self.is_end = False
        else:
            assert x >= 3 or y >= 3
            self.reward = -1
            self.is_end = True


    @property
    def legal_actions(self):
        return () if self.is_end else GriddlerAction.all_actions

    def get_next_state_from_action(self, action: GriddlerAction) -> GriddlerAction:
        return GriddlerState(x=(self.x + (action == GriddlerAction.right)),
                             y=(self.y + (action == GriddlerAction.down)))

    @staticmethod
    def make_initial() -> GriddlerState:
        return GriddlerState(0, 0)

    def __repr__(self) -> str:
        return (f'{type(self).__name__}({self.x}, {self.y})')

    def _as_tuple(self) -> Tuple:
        return (self.x, self.y)

    def __hash__(self) -> int:
        return hash((type(self),) + self._as_tuple())

    def __eq__(self, other: Any) -> bool:
        return ((type(self) is type(other)) and
                (self._as_tuple() == other._as_tuple()))

    n_neurons = 2

    @functools.lru_cache(maxsize=None)
    def to_neurons(self) -> np.ndarray:
        return np.array((self.x, self.y))


class GriddlerStrategy(gamey.SinglePlayerStrategy):
    State = GriddlerState




class ModelFreeLearningCulture(gamey.ModelFreeLearningCulture, gamey.SinglePlayerCulture):
    def __init__(self, *, strategy) -> None:
        gamey.SinglePlayerCulture.__init__(self, state_type=GriddlerState, strategy=strategy)

class ModelFreeLearningStrategy(GriddlerStrategy, gamey.ModelFreeLearningStrategy):

    @property
    def culture(self):
        try:
            return self._culture
        except AttributeError:
            self._culture = ModelFreeLearningCulture(strategy=self)
            return self._culture

    def get_score(self, n: int = 1_000, state_factory: Optional[Callable] = None,
                  max_length: Optional[int] = None) -> int:
        state_factory = (self.culture.make_initial_state if state_factory is None
                         else state_factory)
        last_states = [None] * n
        for states in self.culture.iterate_games(state_factory() for _ in range(n)):
            for i, state in enumerate(states):
                if state is not None:
                    last_states[i] = state
        return np.mean([last_state.reward for last_state in last_states])


class RandomStrategy(GriddlerStrategy, gamey.RandomStrategy):
    pass

class ModelBasedEpisodicLearningStrategy(GriddlerStrategy,
                                         gamey.ModelBasedEpisodicLearningStrategy):
    pass





def demo(n_training_games: int = 1_000, n_evaluation_games: int = 100) -> None:
    print('Starting Griddler demo.')

    # model_free_learning_strategy.get_score(n=1_000)
    learning_strategies = [
        model_based_episodic_learning_strategy := ModelBasedEpisodicLearningStrategy(),
        single_model_free_learning_strategy := ModelFreeLearningStrategy(gamma=1, n_models=1),
        double_model_free_learning_strategy := ModelFreeLearningStrategy(gamma=1, n_models=2),
    ]
    strategies = [
        RandomStrategy(),
        *learning_strategies,
    ]


    print(f"Let's compare {len(strategies)} Griddler strategies. First we'll play "
          f"{n_evaluation_games:,} games on each strategy and observe the scores:\n")

    def print_summary():
        strategies_and_scores = sorted(
            ((strategy, strategy.get_score(n_evaluation_games)) for strategy in strategies),
            key=lambda x: x[1], reverse=True
        )
        for strategy, score in strategies_and_scores:
            print(f'    {strategy}: '.ljust(60), end='')
            print(f'{score: .3f}')

    print_summary()

    print(f"\nThat's nice. Now we want to see that the learning strategies can be better than "
          f"the dumb ones, if we give them time to learn.")

    print(f'Training {model_based_episodic_learning_strategy} on {n_training_games:,} games... ',
          end='')
    sys.stdout.flush()
    model_based_episodic_learning_strategy.get_score(n=n_training_games)
    print('Done.')

    for model_free_learning_strategy in (single_model_free_learning_strategy,
                                         double_model_free_learning_strategy):
        print(f'Training {model_free_learning_strategy} on {n_training_games:,} games',
              end='')
        sys.stdout.flush()
        trainer = model_free_learning_strategy.culture.multi_game_train(
                                               n_total_games=n_training_games, n_games_per_phase=10)
        for _ in more_itertools.chunked(trainer, 10):
            print('.', end='')
        print(' Done.')

    print("\nNow let's run the old comparison again, and see what's the new score for the "
          "learning strategies:\n")

    print_summary()



if __name__ == '__main__':
    demo()

