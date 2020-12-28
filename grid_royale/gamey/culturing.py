# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import itertools
import collections

import more_itertools

from .utils import ImmutableDict
from .base import State, PlayerId, SinglePlayerState
from . import exceptions
from . import policing

class Culture:
    def __init__(self, state_type: Type[State],
                 player_id_to_policy: Mapping[PlayerId, policing.Policy]) -> None:
        self.State = state_type
        self.player_id_to_policy = player_id_to_policy

    @property
    def policy_to_player_ids(self):
        policy_to_player_ids = collections.defaultdict(list)
        for player_id, policy in self.player_id_to_policy.items():
            policy_to_player_ids[policy].append(player_id)
        return {policy: tuple(player_ids) for policy, player_ids
                in policy_to_player_ids.items()}


    def iterate_many_games(self, *, n: int = 10, max_length: int = 100,
                           state_factory: Optional[Callable] = None, be_training: bool = True) \
                                                                                 -> Iterator[State]:
        # Todo: Can this be combined with `iterate_games`?
        state_factory = ((lambda: self.make_initial_state()) if state_factory is None
                         else state_factory)
        for _ in range(n):
            state: State = state_factory()
            yield from self.iterate_game(state, max_length, be_training=be_training)


    def iterate_game(self, state: State, max_length: Optional[int] = None, *,
                     be_training: bool = True) -> Iterator[State]:
        yield state
        iterator = range(1, max_length) if max_length is not None else itertools.count(1)
        for _ in iterator:
            if state.is_end:
                return
            state = self.get_next_state(state, be_training=be_training)
            yield state




class SinglePlayerCulture(Culture):

    def __init__(self, state_type: Type[SoloState], *,
                 policy: policing.Policy) -> None:
        self.policy = policy
        Culture.__init__(self, state_type=state_type,
                         player_id_to_policy=ImmutableDict({None: policy}))


class ModelFreeLearningCulture(Culture):
    def get_next_state(self, state: State, *, be_training: bool = True) -> State:
        if state.is_end:
            raise exceptions.GameOver
        (next_state,) = self._get_next_states((state,), be_training=be_training)
        return next_state

    def _get_next_states(self, states: Iterable[Optional[State]], *, be_training: bool = True) \
                                                                          -> Tuple[Optional[State]]:
        from .model_free import ModelFreeLearningPolicy
        states = tuple(states)
        # policy_to_player_ids = self.policy_to_player_ids
        policy_to_observations = collections.defaultdict(list)
        for state in states:
            if state is None or state.is_end:
                continue
            for player_id, observation in state.player_id_to_observation.items():
                policy_to_observations[self.player_id_to_policy[player_id]].append(observation)

        for policy, observations in policy_to_observations.items():
            policy: ModelFreeLearningPolicy
            q_maps = policy.get_qs_for_observations(observations)
            policy.q_map_cache.update(dict(zip(observations, q_maps)))


        next_states = []

        for state in states:
            if state is None or state.is_end:
                next_states.append(None)
                continue
            player_id_to_action = {
                player_id: self.player_id_to_policy[player_id
                                                      ].get_next_action_and_policy(observation)
                for player_id, observation in state.player_id_to_observation.items()
                if not observation.is_end
            }
            next_state = state.get_next_state_from_actions(player_id_to_action)
            next_states.append(next_state)
            if be_training:
                for player_id, action in player_id_to_action.items():
                    policy = self.player_id_to_policy[player_id]
                    observation = state.player_id_to_observation[player_id]
                    policy.train(observation, action,
                                   next_state.player_id_to_observation[player_id])

        return tuple(next_states)


    def iterate_games(self, states: Iterable[State], max_length: Optional[int] = None, *,
                      be_training: bool = True) -> Iterator[State]:
        states = tuple(states)
        yield states
        iterator = range(1, max_length) if max_length is not None else itertools.count(1)
        for _ in iterator:
            states = self._get_next_states(states, be_training=be_training)
            if all((state is None) for state in states):
                return
            yield states

    def multi_game_train(self, n_total_games: int = 100, *, n_games_per_phase: int = 10,
                         max_length: int = None):
        n_games_by_phase = (
            (n_total_games // n_games_per_phase) * (n_games_per_phase,) +
            bool(n_total_games % n_games_per_phase) * (n_total_games % n_games_per_phase,)
        )
        assert sum(n_games_by_phase) == n_total_games
        assert 0 <= len(n_games_by_phase) - (n_total_games // n_games_per_phase) <= 1

        for n_games in n_games_by_phase:
            states = tuple(self.make_initial_state() for _ in range(n_games))
            for states in self.iterate_games(states, max_length=max_length):
                yield states



