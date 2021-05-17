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
        return len(self.game.cultures[0])

    def __iter__(self) -> int:
        return iter(self.game.cultures[0])

    def __getitem__(self, player_id: PlayerId) -> int:
        assert player_id in self.game.cultures[0]
        return Narrative(self.game, player_id)

    def __repr__(self):
        return (f'<{type(self).__name__} with {len(self)} narratives>')

class Game(collections.abc.Sequence):
    def __init__(self, *, states: Iterable[aggregating.State],
                 cultures: Iterable[aggregating.Culture],
                 activities: Iterable[aggregating.Activity],
                 payoffs: Iterable[aggregating.Payoff]) -> None:
        self.states = list(states)
        self.cultures = list(cultures)
        self.activities = list(activities)
        self.payoffs = list(payoffs)
        self.narratives = NarrativeManager(self)
        self._assert_correct_lengths()

    @classmethod
    def from_state_culture(cls, state: aggregating.State, culture: aggregating.Culture) -> Game:
        return cls(states=(state,), cultures=(culture,), activities=(), payoffs=())

    def _assert_correct_lengths(self) -> None:
        assert (len(self.cultures) == len(self.states) ==
                len(self.activities) + 1 == len(self.payoffs) + 1)


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
    def multi_iterate(games: Sequence[Game]) -> Iterator[tuple[Optional[aggregating.State], ...]]:

        for game in games:
            game._assert_correct_lengths()

        finished_game_indices = set()

        def enumerate_unfinished_games():
            for j, game in enumerate(games):
                if j not in finished_game_indices:
                    yield (j, game)


        for i in itertools.count():
            states = [None] * len(games)
            game_indices_to_play = []
            for j, game in enumerate_unfinished_games():
                try:
                    states[j] = game.states[i]
                except IndexError:
                    game_indices_to_play.append(j)


            ###########################################################

            # todo: This is all shit below, gotta replace it:

            # strategy_to_player_ids = self.strategy_to_player_ids
            strategy_to_observations = collections.defaultdict(list)
            for state in states:
                if state is None or state.is_end:
                    continue
                for player_id, observation in state.player_id_to_observation.items():
                    strategy_to_observations[self.player_id_to_strategy[player_id]].append(observation)

            for strategy, observations in strategy_to_observations.items():
                strategy: ModelFreeLearningStrategy
                q_maps = strategy.get_qs_for_observations(observations)
                strategy.q_map_cache.update(dict(zip(observations, q_maps)))

            yield from self.states

            state: aggregating.State = self.states[-1]
            culture: aggregating.Culture = self.cultures[-1]

            while not state.is_end:
                activity = culture.get_next_activity(state)
                self.activities.append(activity)

                payoff, state = state.get_next_payoff_and_state(activity)
                self.payoffs.append(payoff)
                self.states.append(state)

                culture = culture.get_next_culture(self.states[-2], activity, payoff, state)
                self.cultures.append(culture)

                self._assert_correct_lengths()
                yield state

            ###########################################################

            if all((state is None) for state in states):
                assert finished_game_indices == set(range(len(games)))
                return
            yield tuple(states)





