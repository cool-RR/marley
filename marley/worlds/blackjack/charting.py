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
from .sharknadoing import BlackjackProject

def get_chart_html(blackjack_project: BlackjackProject) -> str:
    import altair
    import pandas as pd

    with blackjack_project.parchment_lock:
        raw_data = tuple(
            tuple(game_sequence.mean_score for game_sequence in agent.game_sequences)
            for agent in blackjack_project.agents
        )
        baseline_data_frame = pd.DataFrame(
            (
                (baseline_policy_evaluation.title,
                 baseline_policy_evaluation.game_sequence.mean_score)
                for baseline_policy_evaluation in blackjack_project.baseline_policy_evaluations
            ),
            columns=('name', 'score')
        )

    data_frame = pd.DataFrame(
        (
            (i_policy, generation, score)
            for i_policy, scores in enumerate(raw_data)
            for generation, score in enumerate(scores)
        ),
        columns=('i_policy', 'generation', 'score')
    )


    base_chart = altair.Chart(data_frame)

    base_chart_with_generation = (
        base_chart.
        encode(
            altair.X(
                'generation', type='quantitative',
                axis=altair.Axis(title='generations', tickMinStep=1)
            ),
        )
    )


    score_mean_chart = (
        base_chart_with_generation.
        mark_line().
        encode(
            altair.Y('score', type='quantitative', aggregate='mean')
        )
    )

    score_ci_chart = (
        base_chart_with_generation.
        mark_errorband(extent='ci').
        encode(
            altair.Y('score', type='quantitative')
        )
    )

    baseline_rule_chart = (
        altair.Chart(baseline_data_frame).
        mark_rule().
        encode(
            altair.Y('score')
        )
    )

    baseline_text_chart = (
        altair.Chart(baseline_data_frame).
        mark_text(
            align='left',
            x='width',
            dx=5,
            fontSize=15,
        ).
        encode(
            altair.Y('score'),
            altair.Text('name')
        )
    )


    score_chart = altair.layer(score_mean_chart, score_ci_chart)
    baseline_chart = altair.layer(baseline_rule_chart, baseline_text_chart)

    chart_title = (f'The gradual rise of the score achieved by {len(blackjack_project.agents)} '
                   f'agents with learning policies, compared with a few baseline '
                   f'policies. The shaded area is a bootstrapped 95% confidence interval.')
    chart = (
        altair.layer(score_chart, baseline_chart, title=chart_title).
        properties(width=800, height=600)
    )


    string_io = io.StringIO()
    chart.save(string_io, format='html')
    return string_io.getvalue()

