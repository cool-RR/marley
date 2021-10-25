# Copyright 2020 Ram Rachum and collaborators.,
# This program is distributed under the MIT license.

from __future__ import annotations
from typing import Iterable, Tuple, Any

import sys
import itertools
import pathlib
import random
import enum
import functools
import io
import numbers
import logging
from typing import Optional, Union
import contextlib
import shlex

import numpy as np
import click
import more_itertools
import numpy as np

from marley import gamey
from marley import jamswank
from marley import sharknado
from .core import *

N_GAMES_PER_SEQUENCE = 300
N_GAMES_PER_BASELINE_POLICY_EVALUATION = 3_000

BASELINE_POLICY_TYPES_AND_ARGS = (
    (AlwaysStickPolicy, ()),
    (ThresholdPolicy, (15,)),
    (ThresholdPolicy, (16,)),
)


class BlackjackProject(gamey.GameySwank):
    def get_job(self, n_generations: int = 3, n_agents: int = 2) -> BlackjackJob:
        return BlackjackJob(self, n_generations=n_generations, n_agents=n_agents)

    agents = jamswank.ParchmentField()
    baseline_policy_evaluations = jamswank.ParchmentField()


    def get_chart_html(self) -> str:
        from .charting import get_chart_html
        return get_chart_html(self)

    def write_chart_to_desktop(self) -> None:
        chart_path = pathlib.Path.home() / 'Desktop' / 'chart.html'
        chart_path.write_text(self.get_chart_html())
        logger.info(f'Wrote chart to {shlex.quote(str(chart_path))}')



class Agent(gamey.GameySwank):
    policies = jamswank.ParchmentField()
    game_sequences = jamswank.ParchmentField()


class GameSequence(gamey.GameySwank):
    games = jamswank.ParchmentField()
    mean_score = jamswank.SimpleField()

    def defrag(self):
        first_game = self.games[0]
        # swank_database = first_game.swank_database
        parchment_field_names = ('states', 'activities', 'payoffs')
        parchments = {
            parchment_field_name: getattr(first_game, parchment_field_name).jam_parchment
            for parchment_field_name in parchment_field_names
        }
        for parchment_field_name, parchment in parchments.items():
            getattr(first_game, parchment_field_name).end_index = len(parchment)
        first_game.save()
        # culture_jam_item = swank_database.get_jam_item(first_game.culture)
        # assert culture_jam_item.jam_index == 0
        # cultures_parchment: jamswank.JamParchment = culture_jam_item.jam_parchment
        # assert len(cultures_parchment) == 1

        for i, (left_game, right_game) in enumerate(gamey.utils.iterate_windowed_pairs(self.games)):
            for parchment_field_name in parchment_field_names:
                left_game_parchment_ref = getattr(left_game, parchment_field_name)
                right_game_parchment_ref = getattr(right_game, parchment_field_name)
                right_game_parchment_ref.migrate(
                    left_game_parchment_ref.jam_id,
                    left_game_parchment_ref.end_index,
                    left_game_parchment_ref.end_index + len(right_game_parchment_ref),
                )
            # old_culture_parchment = swank_database.get_jam_item(right_game.culture).jam_parchment
            # right_game.culture.jam_id = cultures_parchment.jam_id
            # right_game.culture.jam_index = i + 1
            right_game.save()
            # old_culture_parchment.delete()



class BaselinePolicyEvaluation(gamey.GameySwank):
    game_sequence = jamswank.SwankField()
    title = jamswank.SimpleField()



class JobMixin:

    def __init__(self,
                 blackjack_project_or_ref: Union[BlackjackProject, jamswank.SwankRef]) -> None:
        self.blackjack_project_ref = jamswank.SwankRef.from_swank_or_ref(blackjack_project_or_ref)

    def _reduce(self) -> tuple:
        return (type(self), self.blackjack_project_ref)

    def get_agent_ref(self, i_agent: int) -> jamswank.SwankRef:
        with self.blackjack_project_ref.lock_and_load() as blackjack_project:
            try:
                agent = blackjack_project.agents[i_agent]
            except IndexError:
                blackjack_project.agents[i_agent] = agent = Agent()
                agent.save()
                blackjack_project.save()
            return jamswank.SwankRef.from_swank_or_ref(agent)




class BlackjackJob(JobMixin, sharknado.ThinJob):
    def __init__(self, blackjack_project: BlackjackProject, n_generations: int = 5,
                 n_agents: int = 4) -> None:
        JobMixin.__init__(self, blackjack_project)
        self.n_generations = n_generations
        self.n_agents = n_agents

    def _reduce(self) -> tuple:
        return (type(self), self.blackjack_project_ref, self.n_generations, self.n_agents)

    def get_parent_job_to_weight(self):
        return {
            AgentJob(self.blackjack_project_ref, n_generations=self.n_generations):
                                                             sharknado.TadpoleWeight(self.n_agents),
            EvaluateBaselinePolicyJob(self.blackjack_project_ref):
                                       sharknado.TadpoleWeight(len(BASELINE_POLICY_TYPES_AND_ARGS)),
        }

    def thin_run(self):
        with self.blackjack_project_ref.lock_and_load() as blackjack_project:
            blackjack_project.write_chart_to_desktop()



class AgentJob(JobMixin, sharknado.ParallelJob):
    dimensions = 1
    def __init__(self,
                 blackjack_project_or_ref: Union[BlackjackProject, jamswank.SwankRef], *,
                 n_generations: int) -> None:
        JobMixin.__init__(self, blackjack_project_or_ref)
        self.n_generations = n_generations


    def get_parent_job_to_weight(self):
        return {
            PolicyJob(self.blackjack_project_ref):
                sharknado.TadpoleWeight(self.n_generations),
            AgentGameSequenceJob(self.blackjack_project_ref):
                sharknado.TadpoleWeight(self.n_generations),
        }


class AgentGameSequenceJob(JobMixin, sharknado.ParallelJob):
    dimensions = 2

    def fat_sniff(self, fat_gain: sharknado.FatGain) -> sharknado.IntCrowd:
        agent_int_crowds = fat_gain.int_crowd.separate((0,))

        i_agents = tuple(more_itertools.first(agent_int_crowd)[0]
                         for agent_int_crowd in agent_int_crowds)
        if not i_agents:
            return
        i_to_agent = {}
        result_int_crowds = []

        with self.blackjack_project_ref.lock_and_load() as blackjack_project:
            for i_agent in i_agents:
                try:
                    agent = blackjack_project.agents[i_agent]
                except IndexError:
                    continue
                else:
                    i_to_agent[i_agent] = agent

        if i_to_agent:
            arbitrary_agent = more_itertools.first(i_to_agent.values())
            with arbitrary_agent.parchment_lock:
                for i_agent, agent_int_crowd in zip(i_agents, agent_int_crowds):
                    try:
                        agent = i_to_agent[i_agent]
                    except KeyError:
                        continue
                    result_int_crowds.append(
                        sharknado.IntCrowd(
                            sharknado.Point((i_agent, i_game_sequence))
                            for (i_agent, i_game_sequence) in agent_int_crowd
                            if agent.game_sequences.has_index(i_game_sequence)
                        )
                    )

        return sharknado.utils.union(result_int_crowds) if result_int_crowds else None

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        (i_agent, i_policy) = fat_gain.int_crowd.get_single_point()
        agent_ref = self.get_agent_ref(i_agent)
        with agent_ref.lock_and_load() as agent:
            policy: ModelFreeLearningPolicy = agent.policies[i_policy]
        game_sequence = GameSequence()
        game_sequence.save() # Saving it separately, so we add it to the project only after it's
                             # done, and never have to think about partially-written game
                             # sequencees.
        culture = policy.make_culture()
        culture.save()
        for _ in range(N_GAMES_PER_SEQUENCE):
            game = gamey.Game.from_state_culture(BlackjackState.make_initial(), culture)
            game_sequence.games.append(game)

        gamey.Game.multi_crunch(game_sequence.games)

        game_sequence.mean_score = np.mean(
            tuple(
                sum(payoff.get_single() for payoff in game.payoffs) for game in game_sequence.games
            )
        )
        game_sequence.save()
        game_sequence.defrag()
        with agent_ref.lock_and_load(save=True) as agent:
            agent.game_sequences[i_policy] = game_sequence

    def get_parent_job_to_weight(self):
        return {PolicyJob(self.blackjack_project_ref): sharknado.CalfWeight()}


class PolicyJob(JobMixin, sharknado.SerialJob):
    dimensions = 2

    def fat_sniff(self, fat_gain: sharknado.FatGain) -> sharknado.IntCrowd:
        agent_int_crowds = fat_gain.int_crowd.separate((0,))

        i_agents = tuple(more_itertools.first(agent_int_crowd)[0]
                         for agent_int_crowd in agent_int_crowds)
        if not i_agents:
            return
        i_to_agent = {}
        result_int_crowds = []

        with self.blackjack_project_ref.lock_and_load() as blackjack_project:
            for i_agent in i_agents:
                try:
                    agent = blackjack_project.agents[i_agent]
                except IndexError:
                    continue
                else:
                    i_to_agent[i_agent] = agent

        if i_to_agent:
            arbitrary_agent = more_itertools.first(i_to_agent.values())
            with arbitrary_agent.parchment_lock:
                for i_agent, agent_int_crowd in zip(i_agents, agent_int_crowds):
                    try:
                        agent = i_to_agent[i_agent]
                    except KeyError:
                        continue
                    result_int_crowds.append(
                        sharknado.IntCrowd(
                            sharknado.Point((i_agent, i_policy)) for (i_agent, i_policy)
                            in agent_int_crowd if agent.policies.has_index(i_policy)
                        )
                    )

        return sharknado.utils.union(result_int_crowds) if result_int_crowds else None

    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        (i_agent, i_policy) = fat_gain.int_crowd.get_single_point()
        agent_ref = self.get_agent_ref(i_agent)

        if i_policy == 0:
            policy = ModelFreeLearningPolicy.create(discount=1)
        else:
            with agent_ref.lock_and_load() as agent:
                old_policy: ModelFreeLearningPolicy = agent.policies[i_policy - 1]
                game_sequence = agent.game_sequences[i_policy - 1]
            policy = old_policy.train(tuple(game.narratives[None] for game in game_sequence.games),
                                      n_epochs=5)

        with agent_ref.lock_and_load(save=True) as agent:
            agent.policies[i_policy] = policy

    def get_parent_job_to_weight(self):
        return {AgentGameSequenceJob(self.blackjack_project_ref): sharknado.CalfWeight(-1)}


class EvaluateBaselinePolicyJob(JobMixin, sharknado.ParallelJob):
    dimensions = 1

    def fat_sniff(self, fat_gain: sharknado.FatGain) -> sharknado.IntCrowd:
        policy_indicies = tuple(itertools.chain.from_iterable(fat_gain.int_crowd))
        assert all((0 <= policy_index < len(BASELINE_POLICY_TYPES_AND_ARGS))
                   for policy_index in policy_indicies)
        with self.blackjack_project_ref.lock_and_load() as blackjack_project:
            finished_policy_indices = tuple(
                filter(blackjack_project.baseline_policy_evaluations.has_index, policy_indicies)
            )
        return sharknado.IntCrowd(sharknado.Point((finished_policy_index,)) for
                                  finished_policy_index in finished_policy_indices)


    def fat_run(self, fat_gain: sharknado.FatGain) -> None:
        (policy_index,) = fat_gain.int_crowd.get_single_point()
        (baseline_policy_type, baseline_policy_args) = BASELINE_POLICY_TYPES_AND_ARGS[policy_index]
        baseline_policy: BlackjackPolicy = baseline_policy_type(*baseline_policy_args)

        game_sequence = GameSequence()
        game_sequence.save()

        baseline_policy_evaluation = BaselinePolicyEvaluation(
            title=baseline_policy.title,
            game_sequence=game_sequence,
        )
        baseline_policy_evaluation.save()

        culture = baseline_policy.make_culture()
        culture.save()
        for _ in range(N_GAMES_PER_BASELINE_POLICY_EVALUATION):
            game = gamey.Game.from_state_culture(BlackjackState.make_initial(), culture)
            game_sequence.games.append(game)

        gamey.Game.multi_crunch(game_sequence.games)

        game_sequence.mean_score = np.mean(
            tuple(
                sum(payoff.get_single() for payoff in game.payoffs)
                for game in game_sequence.games
            )
        )
        game_sequence.save()
        game_sequence.defrag()
        with self.blackjack_project_ref.lock_and_load(save=True) as blackjack_project:
            blackjack_project: BlackjackProject
            blackjack_project.baseline_policy_evaluations[policy_index] = baseline_policy_evaluation
