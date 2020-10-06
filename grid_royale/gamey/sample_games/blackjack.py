# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import math
import sys
import inspect
import abc
import random
import itertools
import collections.abc
import statistics
import concurrent.futures
import enum
import functools
import numbers
from typing import Iterable, Union, Optional, Tuple, Any, Iterator
import dataclasses

import more_itertools
import numpy as np

from grid_royale import gamey


def sum_cards(cards: Iterable[int]) -> int:
    result = 0
    n_aces = 0
    for card in cards:
        if 2 <= card <= 10:
            result += card
        else:
            assert card == 1
            n_aces += 1
            result += 11

    if result > 21: # Handling the aces:
        for _ in range(n_aces):
            result -= 10
            if result <= 21:
                break

    return result

assert sum_cards((1, 2, 3)) == 16
assert sum_cards((1, 2, 3, 5)) == 21
assert sum_cards((1, 2, 3, 6)) == 12


class _BlackjackActionType(type(gamey.Action), type(enum.Enum)):
    pass


class BlackjackAction(gamey.Action, enum.Enum, metaclass=_BlackjackActionType):
    hit = 'hit'
    stick = 'stick'
    wait = 'wait'

BlackjackAction.all_actions = (BlackjackAction.hit, BlackjackAction.stick,
                               BlackjackAction.wait)

_card_distribution = tuple(range(1, 10 + 1)) + (10,) * 3
def get_random_card() -> int:
    return random.choice(_card_distribution)

class BlackjackState(gamey.SinglePlayerState):

    Action = BlackjackAction

    reward = 0

    def __init__(self, player_cards: Tuple[int, ...], dealer_cards: Tuple[int, ...]) -> None:
        self.player_cards = tuple(sorted(player_cards))
        self.dealer_cards = tuple(sorted(dealer_cards))

        self.player_stuck = (len(self.dealer_cards) >= 2)
        self.player_sum = sum_cards(self.player_cards)
        self.dealer_sum = sum_cards(self.dealer_cards)

        ### Calculating end value, if any: ####################################
        #                                                                     #
        if self.dealer_sum > 21:
            self.is_end = True
            self.reward = 1
        elif self.dealer_sum == 21:
            self.is_end = True
            assert self.player_sum <= 21
            self.reward = 0 if self.player_sum == 21 else -1
        elif 17 <= self.dealer_sum <= 20:
            assert self.player_stuck
            self.is_end = True
            if self.player_sum > self.dealer_sum:
                self.reward = 1
            elif self.player_sum < self.dealer_sum:
                self.reward = -1
            else:
                assert self.player_sum == self.dealer_sum
                self.reward = 0
        else: # len(self.dealer_cards) == 1
            assert 2 <= self.dealer_sum <= 16
            if self.player_stuck:
                self.is_end = False
                assert self.player_sum <= 20
            else: # not self.player_stuck
                if self.player_sum > 21:
                    self.is_end = True
                    self.reward = -1
                elif self.player_sum == 21:
                    self.is_end = True
                    self.reward = 1
                else:
                    assert self.player_sum <= 20
                    self.is_end = False
        #                                                                     #
        ### Finished calculating end value, if any. ###########################

        if self.is_end:
            self.legal_actions = ()
        elif self.player_stuck:
            self.legal_actions = (BlackjackAction.wait,)
        else:
            self.legal_actions = (BlackjackAction.hit, BlackjackAction.stick,)



    def get_next_state_from_action(self, action: BlackjackAction) -> BlackjackState:
        if action not in self.legal_actions:
            raise gamey.exceptions.IllegalAction(action)
        if self.player_stuck or action == BlackjackAction.stick:
            return BlackjackState(
                self.player_cards,
                self.dealer_cards + (get_random_card(),)
            )
        else:
            return BlackjackState(
                self.player_cards + (get_random_card(),),
                self.dealer_cards
            )

    @staticmethod
    def make_initial() -> BlackjackState:
        return BlackjackState(
            (get_random_card(), get_random_card()),
            (get_random_card(),)
        )

    def __repr__(self) -> str:
        return (f'{type(self).__name__}'
                f'({self.player_cards}, {self.dealer_cards})')

    def _as_tuple(self) -> Tuple:
        return (self.player_cards, self.dealer_cards)

    def __hash__(self) -> int:
        return hash(self._as_tuple())

    def __eq__(self, other: Any) -> bool:
        return ((type(self) is type(other)) and
                (self._as_tuple() == other._as_tuple()))

    n_neurons = 5

    @functools.lru_cache()
    def to_neurons(self) -> np.ndarray:
        return np.array(
            tuple((
                self.player_sum / 21,
                1 in self.player_cards,
                self.dealer_sum / 21,
                1 in self.dealer_cards,
                float(self.player_stuck)
            ))
        )



class BlackjackStrategy(gamey.SinglePlayerStrategy):
    State = BlackjackState


class AlwaysHitStrategy(BlackjackStrategy):
    def decide_action_for_observation(self, observation: BlackjackState,
                                       extra: Any = None) -> BlackjackAction:
        return (BlackjackAction.hit if (BlackjackAction.hit in observation.legal_actions)
                else BlackjackAction.wait)

class AlwaysStickStrategy(BlackjackStrategy):
    def decide_action_for_observation(self, observation: BlackjackState,
                                       extra: Any = None) -> BlackjackAction:
        return (BlackjackAction.stick if (BlackjackAction.stick in observation.legal_actions)
                else BlackjackAction.wait)

class ThresholdStrategy(BlackjackStrategy):
    def __init__(self, threshold: int = 17) -> None:
        self.threshold = threshold

    def decide_action_for_observation(self, observation: BlackjackState,
                                       extra: Any = None) -> BlackjackAction:
        if BlackjackAction.wait in observation.legal_actions:
            return BlackjackAction.wait
        elif observation.player_sum >= self.threshold:
            return BlackjackAction.stick
        else:
            return BlackjackAction.hit

    def _extra_repr(self):
        return f'(threshold={self.threshold})'


class RandomStrategy(BlackjackStrategy, gamey.RandomStrategy):
    pass

class ModelBasedEpisodicLearningStrategy(BlackjackStrategy,
                                         gamey.ModelBasedEpisodicLearningStrategy):
    pass

class ModelFreeLearningStrategy(BlackjackStrategy, gamey.ModelFreeLearningStrategy):
    pass



def demo(n_training_games: int) -> None:
    print('Starting Blackjack demo.')

    # model_free_learning_strategy.get_score(n=1_000)
    learning_strategies = [
        ModelBasedEpisodicLearningStrategy(),
        ModelFreeLearningStrategy(gamma=1),
    ]
    strategies = [
        RandomStrategy(),
        AlwaysHitStrategy(),
        AlwaysStickStrategy(),
        ThresholdStrategy(15),
        ThresholdStrategy(16),
        ThresholdStrategy(17),
        *learning_strategies,
    ]

    print(f"Let's compare {len(strategies)} Blackjack strategies. First we'll play 500 games "
          f"on each strategy and observe the scores:\n")

    def print_summary():
        strategies_and_scores = sorted(
            ((strategy, strategy.get_score(500)) for strategy in strategies),
            key=lambda x: x[1], reverse=True
        )
        for strategy, score in strategies_and_scores:
            print(f'    {strategy}: '.ljust(40), end='')
            print(score)

    print_summary()

    print(f"\nThat's nice. Now we want to see that the learning strategies can be better than "
          f"the dumb ones, if we give them time to learn. Let's play {n_training_games:,} "
          "games on each of the two learning strategies.\n")

    for learning_strategy in learning_strategies:
        learning_strategy: gamey.Strategy
        print(f'Training {learning_strategy} on {n_training_games:,} games... ', end='')
        sys.stdout.flush()
        learning_strategy.get_score(n=n_training_games)
        print('Done.')

    print("\nNow let's run the old comparison again, and see what's the new score for the "
          "learning strategies:\n")

    print_summary()



if __name__ == '__main__':
    if len(sys.argv) >= 2:
        n_training_games = int(sys.argv[1])
    else:
        n_training_games = 1_000
    demo(n_training_games=n_training_games)

