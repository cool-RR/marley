# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''
Gamey is a cute reinforcement learning framework.

I wrote Gamey so I would have a clear separation between generic reinforcement learning architecture
and the specific game of GridRoyale. Gamey defineds states, observations, actions, policies,
cultures. It defines these as base classes that should be overridden.

See the `sample_games/blackjack.py` module for an example of using Gamey on Blackjack.
'''

from __future__ import annotations

from typing import Optional

from .base import Observation, Action, ActionEnum, Story, PlayerId
from .policing import Policy, RandomPolicy, SoloEpisodicPolicy, QPolicy
from .aggregating import State, SoloState, Payoff, Activity, Culture
from .gaming import Game, Narrative
from .model_free import ModelFreeLearningPolicy
from .model_based import ModelBasedEpisodicLearningPolicy
from . import exceptions
from marley.gamey.gamey_swanking import GameySwankDatabase, GameySwank


global_gamey_swank_database: GameySwankDatabase = GameySwankDatabase.create_ethereal()
