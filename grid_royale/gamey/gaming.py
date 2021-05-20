# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import math
import inspect
import re
import abc
import random
import itertools
import collections.abc
import statistics
import concurrent.futures
import enum
import functools
import numbers
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import dataclasses

import more_itertools
import numpy as np

from .utils import ImmutableDict
from . import utils
from . import exceptions
from . import aggregating
from .base import PlayerId, Story


class Narrative(collections.abc.Sequence):
    # Todo: same overlap between this class and timeline
    def __init__(self, game: Game, player_id: PlayerId) -> None:
        self.game = game
        self.player_id = player_id

    def __len__(self) -> int:
        return len(self.game) - 1

    def __getitem__(self, i: Union[int, slice]) -> int:
        if isinstance(i, slice):
            raise NotImplementedError
        assert isinstance(i, int)
        if - (len(self) + 1) < i < 0:
            i += len(self)
        if not 0 <= i <= len(self) - 1:
            raise IndexError
        return Story(
            old_observation=self.game.states[i][self.player_id],
            action=self.game.activities[i][self.player_id],
            reward=self.game.payoffs[i][self.player_id],
            new_observation=self.game.states[i + 1][self.player_id],
        )

    def __repr__(self):
        return (f'<{type(self).__name__} with {len(self)} stories>')


class NarrativeManager(collections.abc.Mapping):
    def __init__(self, game: Game) -> None:
        self.game = game

    def __len__(self) -> int:
        return len(self.game.culture)

    def __iter__(self) -> int:
        return iter(self.game.culture)

    def __getitem__(self, player_id: PlayerId) -> int:
        assert player_id in self.game.culture
        return Narrative(self.game, player_id)

    def __repr__(self):
        return (f'<{type(self).__name__} with {len(self)} narratives>')

class Game(collections.abc.Sequence):
    def __init__(self, culture: aggregating.Culture, *, states: Iterable[aggregating.State],
                 activities: Iterable[aggregating.Activity],
                 payoffs: Iterable[aggregating.Payoff]) -> None:
        self.culture = culture
        self.states = list(states)
        self.activities = list(activities)
        self.payoffs = list(payoffs)
        self.narratives = NarrativeManager(self)
        self._assert_correct_lengths()

    @classmethod
    def from_state_culture(cls, state: aggregating.State, culture: aggregating.Culture) -> Game:
        return cls(states=(state,), cultures=(culture,), activities=(), payoffs=())

    def _assert_correct_lengths(self) -> None:
        assert (len(self.states) == len(self.activities) + 1 == len(self.payoffs) + 1)


    def __iter__(self) -> Iterator[aggregating.State]:
        for states in Game.multi_iterate((self,)):
            (state,) = states
            yield state

    def __len__(self) -> int:
        return len(self.states)

    def __repr__(self) -> str:
        return f'<{type(self).__name__}: len={len(self)}>'


    def __getitem__(self, i: Union[int, slice]) -> aggregating.State:
        if isinstance(i, slice):
            raise NotImplementedError
        else:
            assert isinstance(i, int)
            return self.states[i]



    def crunch(self, n: Optional[int] = None) -> None:
        for _ in more_itertools.islice_extended(self)[:n]:
            pass
        assert self.states[-1].is_end or len(self.states) == n

    @staticmethod
    def multi_crunch(games: Sequence[Game], n: Optional[int] = None) -> None:
        for _ in more_itertools.islice_extended(Game.multi_iterate(games))[:n]:
            pass


    @staticmethod
    def multi_iterate(games: Sequence[Game]) -> Iterator[tuple[Optional[aggregating.State], ...]]:
        from . import policing

        for game in games:
            game._assert_correct_lengths()

        finished_game_indices = set()

        def enumerate_unfinished_games():
            for j, game in enumerate(games):
                if j not in finished_game_indices:
                    yield (j, game)


        for i in itertools.count():
            state_deck = [None] * len(games)
            game_indices_to_play = []
            for j, game in enumerate_unfinished_games():
                try:
                    state_deck[j] = game.states[i]
                except IndexError:
                    game_indices_to_play.append(j)


            ###########################################################

            # todo: This is all shit below, gotta replace it:

            games_to_play = tuple(games[j] for j in game_indices_to_play)

            ### Making Q-policies way faster by batching matrix operations: ########################
            #                                                                                      #
            # This section is an optimization, it's a no-op when given policies that don't support
            # it.

            q_policy_to_observations = collections.defaultdict(list)
            iterator = itertools.chain.from_iterable(((game.states[-1] + game.cultures[-1]).values()
                                                      for game in games_to_play))
            for observation, policy in iterator:
                if isinstance(policy, policing.QPolicy):
                    q_policy_to_observations[policy].append(observation)

            for q_policy, observations in q_policy_to_observations.items():
                q_policy.get_qs_for_observations(observations) # Automatically caches the result.

            #                                                                                      #
            ### Finished making Q-policies way faster by batching matrix operations. ###############

            for j, game in zip(game_indices_to_play, games_to_play):
                assert state_deck[j] is None
                state: aggregating.State = game.states[-1]

                if state.is_end:
                    finished_game_indices.add(j)
                    continue

                activity = game.culture.get_next_activity(state)
                game.activities.append(activity)

                next_payoff, next_state = state.get_next_payoff_and_state(activity)
                game.payoffs.append(next_payoff)
                game.states.append(next_state)

                state_deck[j] = next_state

                game._assert_correct_lengths()


            if all((state is None) for state in state_deck):
                assert finished_game_indices == set(range(len(games)))
                return
            yield tuple(state_deck)


    @property
    def total_payoff(self, *, start: int = 0, discount: numbers.Real = 1):
        assert start >= 0
        assert 0 <= discount <= 1
        wip_payoff = collections.defaultdict(lambda: 0)
        payoffs = more_itertools.islice_extended(self.payoffs)[start:]
        for i, payoff in enumerate(payoffs):
            for player_id, reward in payoff.items():
                wip_payoff[player_id] += (discount ** i) * reward

        return aggregating.Payoff(wip_payoff)



