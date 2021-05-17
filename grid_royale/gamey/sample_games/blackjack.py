# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations
from typing import Iterable, Tuple, Any

import sys
import itertools
import random
import enum
import functools
import numbers
import numpy as np

import more_itertools

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

def get_shuffled_deck() -> tuple:
    return tuple(gamey.utils.shuffled(_card_distribution * 4))


class BlackjackState(gamey.SoloState):

    def __init__(self, player_cards: Tuple[int, ...], dealer_cards: Tuple[int, ...],
                 deck: Tuple[int, ...]) -> None:
        gamey.SoloState.__init__(self)
        self.player_cards = tuple(sorted(player_cards))
        self.dealer_cards = tuple(sorted(dealer_cards))
        self.deck = deck

        self.player_stuck = (len(self.dealer_cards) >= 2)
        self.is_first_state = (len(self.player_cards) == len(self.dealer_cards) == 0)
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
        elif 2 <= self.dealer_sum <= 16:
            if self.player_stuck:
                self.is_end = False
                self.reward = 0
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
                    self.reward = 0
        else:
            assert self.is_first_state
            self.is_end = False
            self.reward = 0

        #                                                                     #
        ### Finished calculating end value, if any. ###########################

        if self.is_end:
            self.legal_actions = ()
        elif self.player_stuck:
            self.legal_actions = (BlackjackAction.wait,)
        elif self.is_first_state:
            self.legal_actions = (BlackjackAction.wait,)
        else:
            self.legal_actions = (BlackjackAction.hit, BlackjackAction.stick,)



    def get_next_reward_and_state(self, action: BlackjackAction) -> Tuple[numbers.Number,
                                                                          BlackjackState]:
        if action not in self.legal_actions:
            raise gamey.exceptions.IllegalAction(action)
        if self.is_first_state:
            state = BlackjackState(
                self.deck[-2:],
                (self.deck[-3],),
                self.deck[:-3]
            )
        elif self.player_stuck or action == BlackjackAction.stick:
            state = BlackjackState(
                self.player_cards,
                self.dealer_cards + self.deck[-1:],
                self.deck[:-1]
            )
        else:
            state = BlackjackState(
                self.player_cards + self.deck[-1:],
                self.dealer_cards,
                self.deck[:-1]
            )
        return (state.reward, state)

    @staticmethod
    def make_initial() -> BlackjackState:
        return BlackjackState((), (), get_shuffled_deck())

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

    neural_dtype = np.dtype([('sequential', np.float64, 5)])

    @functools.lru_cache(maxsize=None)
    def to_neural(self) -> np.ndarray:
        sequential_array = np.array(
            tuple((
                self.player_sum / 21,
                1 in self.player_cards,
                self.dealer_sum / 21,
                1 in self.dealer_cards,
                float(self.player_stuck)
            ))
        )
        array = np.zeros((1,), dtype=self.neural_dtype)
        array['sequential'][0] = sequential_array
        return array



class BlackjackPolicy(gamey.SoloEpisodicPolicy):
    pass





class RandomPolicy(BlackjackPolicy, gamey.RandomPolicy):
    pass

class AlwaysHitPolicy(BlackjackPolicy, gamey.CategoricallyStubbornPolicy):
    def get_next_action(self, observation: BlackjackState) -> BlackjackAction:
        return (BlackjackAction.hit if (BlackjackAction.hit in observation.legal_actions)
                else BlackjackAction.wait)

class AlwaysStickPolicy(BlackjackPolicy, gamey.CategoricallyStubbornPolicy):
    '''A policy that always sticks, no matter what.'''
    def get_next_action(self, observation: BlackjackState) -> BlackjackAction:
        return (BlackjackAction.stick if (BlackjackAction.stick in observation.legal_actions)
                else BlackjackAction.wait)

class ThresholdPolicy(BlackjackPolicy, gamey.CategoricallyStubbornPolicy):
    '''
    A policy that sticks if the sum of cards is below the given threshold.
    '''
    def __init__(self, threshold: int = 17) -> None:
        self.threshold = threshold

    def get_next_action(self, observation: BlackjackState) -> BlackjackAction:
        if BlackjackAction.wait in observation.legal_actions:
            return BlackjackAction.wait
        elif observation.player_sum >= self.threshold:
            return BlackjackAction.stick
        else:
            return BlackjackAction.hit

    def _extra_repr(self):
        return f'(threshold={self.threshold})'



class ModelFreeLearningPolicy(gamey.ModelFreeLearningPolicy, BlackjackPolicy):
    Observation = BlackjackState
    Action = BlackjackAction




def demo(n_training_states: int = 1_000, n_evaluation_games: int = 100) -> None:
    print('Starting Blackjack demo.')

    learning_policies = [
        single_model_free_learning_policy := ModelFreeLearningPolicy(gamma=1, n_models=1),
        double_model_free_learning_policy := ModelFreeLearningPolicy(gamma=1, n_models=2),
    ]
    policies = [
        RandomPolicy(),
        AlwaysHitPolicy(),
        AlwaysStickPolicy(),
        ThresholdPolicy(15),
        ThresholdPolicy(16),
        ThresholdPolicy(17),
        *learning_policies,
    ]


    print(f"Let's compare {len(policies)} Blackjack policies. First we'll play "
          f"{n_evaluation_games:,} games on each policy and observe the scores:\n")

    def print_summary():
        policies_and_scores = sorted(
            ((policy, policy.get_score(BlackjackState.make_initial, n_evaluation_games))
             for policy in policies),
            key=lambda x: x[1], reverse=True
        )
        for policy, score in policies_and_scores:
            print(f'    {policy}: '.ljust(60), end='')
            print(f'{score: .3f}')

    print_summary()

    print(f"\nThat's nice. Now we want to see that the learning policies can be better than "
          f"the dumb ones, if we give them time to learn.")

    for model_free_learning_policy in (single_model_free_learning_policy,
                                       double_model_free_learning_policy):
        model_free_learning_policy: ModelFreeLearningPolicy
        print(f'Training {model_free_learning_policy} on {n_training_states:,} states...',
              end='')
        sys.stdout.flush()

        new_model_free_learning_policy = model_free_learning_policy.train(
                                               BlackjackState.make_initial, n_training_states)
        policies[policies.index(model_free_learning_policy)] = new_model_free_learning_policy
        print(' Done.')

    print("\nNow let's run the old comparison again, and see what's the new score for the "
          "learning policies:\n")

    print_summary()



if __name__ == '__main__':
    try:
        n_training_states = int(sys.argv[1])
    except IndexError:
        demo()
    else:
        demo(n_training_states=n_training_states)



