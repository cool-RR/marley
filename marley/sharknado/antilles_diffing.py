# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import threading
import functools
import collections
import io
import itertools
import random
import operator as operator_module
import uuid as uuid_module

import more_itertools

from .weighting import Weight, CalfWeight, TadpoleWeight
from .gaining import Gain, ThinGain, FatGain, GainArg, GainDyad
from .jobbing import Job, ThinJob, FatJob, JobSniffingJob
from .wedging import Wedge
from . import utils

import networkx as nx

fields = (
    'job_to_finished_gain',
    'job_to_pending_gain',
    'job_to_preexisting_gain',
    'job_to_sniffed_gain',
    'job_to_sniff_pending_gain',
    'job_to_directive_gain',
    'job_to_desired_gain',
    'job_to_available_gain',
)

def subtract_dicts(minuend_dict: dict, subtrahend_dict: dict) -> dict:
    return {job: gain for job in {*minuend_dict.keys(), *subtrahend_dict.keys()}
            if (gain := (minuend_dict[job] - subtrahend_dict[job]))}



class AntillesDiff:
    def __init__(self, old_antilles_snapshot: AntillesSnapshot,
                 new_antilles_snapshot: AntillesSnapshot) -> None:
        assert old_antilles_snapshot.keys() == new_antilles_snapshot.keys()

        self.old_antilles_snapshot = old_antilles_snapshot
        self.new_antilles_snapshot = new_antilles_snapshot

        self.additions = {
            field: d for field in old_antilles_snapshot
             if (d := subtract_dicts(new_antilles_snapshot[field], old_antilles_snapshot[field]))
        }
        self.subtractions = {
            field: d for field in old_antilles_snapshot
             if (d := subtract_dicts(old_antilles_snapshot[field], new_antilles_snapshot[field]))
        }


    def __repr__(self) -> str:
        return (f'<{type(self).__name__}: '
                f'{len(self.additions)} addition{"s" if len(self.additions) else ""} and '
                f'{len(self.subtractions)} subtraction{"s" if len(self.additions) else ""}.>')

    def __bool__(self) -> bool:
        return bool(self.additions or self.subtractions)

    def as_dict(self) -> dict:
        return {'additions': self.additions, 'subtractions': self.subtractions}

    def show(self) -> str:
        string_io = io.StringIO()
        for d, title in ((self.additions, 'Additions:\n'),
                         (self.subtractions, 'Subtractions:\n')):
            string_io.write(title)
            if d:
                for field, change in d.items():
                    string_io.write(f'    {field}:\n')
                    for job, gain in change.items():
                        string_io.write(f'        {job}: {gain.arg}\n')
            else:
                string_io.write(f'    None.\n')

        return string_io.getvalue()





class AntillesDiffRecorder:
    def __init__(self, antilles: Antilles):
        self.antilles = antilles
        self.is_entered = False
        self.old_antilles_snapshot: Optional[AntillesSnapshot] = None
        self.new_antilles_snapshot: Optional[AntillesSnapshot] = None
        self.antilles_diff: Optional[AntillesDiff] = None


    def __enter__(self) -> AntillesDiffRecorder:
        assert self.is_entered is False
        assert self.old_antilles_snapshot is None
        assert self.new_antilles_snapshot is None
        assert self.antilles_diff is None

        self.is_entered = True
        self.old_antilles_snapshot = AntillesSnapshot.create_from_antilles(self.antilles)
        return self


    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        assert self.is_entered is True
        assert self.old_antilles_snapshot is not None
        assert self.new_antilles_snapshot is None
        assert self.antilles_diff is None

        self.is_entered = False
        self.new_antilles_snapshot = AntillesSnapshot.create_from_antilles(self.antilles)
        self.antilles_diff = AntillesDiff(self.old_antilles_snapshot,
                                          self.new_antilles_snapshot)





class AntillesSnapshot(utils.CuteUserDict):
    @classmethod
    def create_from_antilles(cls, antilles: Antilles) -> AntillesSnapshot:
        with antilles.lock:
            return AntillesSnapshot(
                ((field, JobToItsGainDict(getattr(antilles, field))) for field in fields)
            )

    def show(self) -> str:
        string_io = io.StringIO()
        for field, d in self.items():
            string_io.write(f'{field}:\n')
            if d:
                for job, gain in d.items():
                    string_io.write(f'    {job}: {gain.arg}\n')
            else:
                string_io.write(f'    None.\n')

        return string_io.getvalue()







from .antillesing import Antilles, JobToItsGainDict