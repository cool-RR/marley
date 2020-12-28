# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

'''
Gamey is a cute reinforcement learning framework.

I wrote Gamey so I would have a clear separation between generic reinforcement learning architecture
and the specific game of GridRoyale. Gamey defineds states, observations, actions, policies,
cultures. It defines these as base classes that should be overridden.

See the `sample_games/blackjack.py` module for an example of using Gamey on Blackjack.
'''


from .base import Observation, State, SoloState, Action, ActionEnum
from .policing import Policy, RandomPolicy, SinglePlayerPolicy, QPolicy
from .culturing import Culture, SinglePlayerCulture, ModelFreeLearningCulture
from .model_free import ModelFreeLearningPolicy
from .model_based import ModelBasedEpisodicLearningPolicy
from . import exceptions