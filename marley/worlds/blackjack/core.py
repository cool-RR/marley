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
import logging
from typing import Optional, Union

import numpy as np
import click
import more_itertools

from marley import gamey
from marley.jamswank.swanking import SimpleField, SavvyField, SwankDatabase
from marley.jamswank.jamming import JamId

logger = logging.getLogger(__name__)



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
    player_cards = SimpleField()
    dealer_cards = SimpleField()
    deck = SimpleField()

    def __init__(self, *, player_cards: Tuple[int, ...], dealer_cards: Tuple[int, ...],
                 deck: Tuple[int, ...], player_id_to_observation: Optional[dict] = None,
                 jam_id: Optional[JamId] = None, jam_index: Optional[int] = None,
                 swank_database: Optional[SwankDatabase] = None) -> None:
        gamey.SoloState.__init__(self,
                                 player_cards=tuple(sorted(player_cards)),
                                 dealer_cards=tuple(sorted(dealer_cards)),
                                 deck=tuple(deck),
                                 player_id_to_observation=(player_id_to_observation or
                                                           {None: BlackjackObservation(self)}),
                                 jam_id=jam_id,
                                 jam_index=jam_index,
                                 swank_database=swank_database)

        self.player_stuck = (len(self.dealer_cards) >= 2)
        self.is_first_state = (len(self.player_cards) == len(self.dealer_cards) == 0)
        self.player_sum = sum_cards(self.player_cards)
        self.dealer_sum = sum_cards(self.dealer_cards)

        ### Calculating end value, if any: #########################################################
        #                                                                                          #
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

        #                                                                                          #
        ### Finished calculating end value, if any. ################################################

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
                player_cards=self.deck[-2:],
                dealer_cards=(self.deck[-3],),
                deck=self.deck[:-3]
            )
        elif self.player_stuck or action == BlackjackAction.stick:
            state = BlackjackState(
                player_cards=self.player_cards,
                dealer_cards=(self.dealer_cards + self.deck[-1:]),
                deck=self.deck[:-1]
            )
        else:
            state = BlackjackState(
                player_cards=(self.player_cards + self.deck[-1:]),
                dealer_cards=self.dealer_cards,
                deck=self.deck[:-1]
            )
        return (state.reward, state)

    @staticmethod
    def make_initial() -> BlackjackState:
        return BlackjackState(player_cards=(), dealer_cards=(), deck=get_shuffled_deck())

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


class BlackjackObservation(gamey.Observation):
    def __init__(self, state: BlackjackState):
        self.state = state

    def to_savvy_content(self) -> tuple:
        return (self.state,)

    legal_actions = property(lambda self: self.state.legal_actions)
    player_sum = property(lambda self: self.state.player_sum)
    dealer_sum = property(lambda self: self.state.dealer_sum)
    player_cards = property(lambda self: self.state.player_cards)
    dealer_cards = property(lambda self: self.state.dealer_cards)
    player_stuck = property(lambda self: self.state.player_stuck)

    neural_dtype = np.dtype([('sequential_input', np.float64, 5)])

    def _to_neural(self) -> np.ndarray:
        sequential_input_array = np.array(
            tuple((
                self.player_sum / 21,
                1 in self.player_cards,
                self.dealer_sum / 21,
                1 in self.dealer_cards,
                float(self.player_stuck)
            ))
        )
        array = np.zeros((1,), dtype=self.neural_dtype)
        array['sequential_input'][0] = sequential_input_array
        return array




class BlackjackPolicy(gamey.SoloEpisodicPolicy):
    pass





class RandomPolicy(BlackjackPolicy, gamey.RandomPolicy):
    pass

class AlwaysHitPolicy(BlackjackPolicy):
    def get_next_action(self, observation: BlackjackState) -> BlackjackAction:
        return (BlackjackAction.hit if (BlackjackAction.hit in observation.legal_actions)
                else BlackjackAction.wait)

class AlwaysStickPolicy(BlackjackPolicy):
    '''A policy that always sticks, no matter what.'''
    def get_next_action(self, observation: BlackjackState) -> BlackjackAction:
        return (BlackjackAction.stick if (BlackjackAction.stick in observation.legal_actions)
                else BlackjackAction.wait)

    title = 'Always stick'

class ThresholdPolicy(BlackjackPolicy):
    '''
    A policy that sticks if the sum of cards is below the given threshold.
    '''
    def __init__(self, threshold: int = 17, *args, **kwargs) -> None:
        BlackjackPolicy.__init__(self, *args, **kwargs)
        self.threshold = threshold
        self.title = f'Stick at {threshold}+'


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
    observation_type = BlackjackObservation
    action_type: Type[BlackjackAction] = SavvyField(lambda: BlackjackAction)
    default_block_size = 2_000_000





