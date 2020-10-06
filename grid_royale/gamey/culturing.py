# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar)
import itertools

from .utils import ImmutableDict
from .base import State, PlayerId, SinglePlayerState
from . import exceptions
from . import strategizing

class Culture:
    def __init__(self, state_type: Type[State],
                 player_id_to_strategy: Mapping[PlayerId, strategizing.Strategy]) -> None:
        self.State = state_type
        self.player_id_to_strategy = player_id_to_strategy


    def iterate_many_games(self, *, n: int = 10, max_length: int = 100,
                           state_factory: Optional[Callable] = None, be_training: bool = True) \
                                                                                 -> Iterator[State]:
        state_factory = ((lambda: self.State.make_initial()) if state_factory is None
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


    def get_next_state(self, state: State, *, be_training: bool = True) -> State:
        if state.is_end:
            raise exceptions.GameOver
        player_id_to_action = {
            player_id: self.player_id_to_strategy[player_id
                                                        ].decide_action_for_observation(observation)
            for player_id, observation in state.player_id_to_observation.items()
            if not observation.is_end
        }
        next_state = state.get_next_state_from_actions(player_id_to_action)
        if be_training:
            for player_id, action in player_id_to_action.items():
                strategy = self.player_id_to_strategy[player_id]
                observation = state.player_id_to_observation[player_id]
                strategy.train(observation, action, next_state.player_id_to_observation[player_id])
        return next_state


class SinglePlayerCulture(Culture):

    def __init__(self, state_type: Type[SinglePlayerState], *,
                 strategy: strategizing.Strategy) -> None:
        self.strategy = strategy
        Culture.__init__(self, state_type=state_type,
                         player_id_to_strategy=ImmutableDict({None: strategy}))

