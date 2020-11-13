# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''
Gamey is a cute reinforcement learning framework.

I wrote Gamey so I would have a clear separation between generic reinforcement learning architecture
and the specific game of GridRoyale. Gamey defineds states, observations, actions, strategies,
cultures. It defines these as base classes that should be overridden.

See the `sample_games/blackjack.py` module for an example of using Gamey on Blackjack.
'''


from .base import Observation, State, SinglePlayerState, Action, ActionEnum, ActionObservation
from .strategizing import Strategy, RandomStrategy, SinglePlayerStrategy, QStrategy
from .culturing import Culture, SinglePlayerCulture, ModelFreeLearningCulture
from .model_free import ModelFreeLearningStrategy
from .model_based import ModelBasedEpisodicLearningStrategy
from . import exceptions