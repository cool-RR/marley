# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type, FrozenSet,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict, Set)
import itertools
import collections
import dataclasses

import networkx as nx

from .jobbing import Job
from .weighting import Weight, CalfWeight, TadpoleWeight

@dataclasses.dataclass(order=True, frozen=True)
class Relationship:
    parent_job: Job
    child_job: Job
    weight: Weight



class Wedge:
    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    def _add_edge(self, parent_job: Job, child_job: Job,
                 weight: Union[CalfWeight, TadpoleWeight]) -> None:
        from .jobbing import ThinJob, FatJob
        assert isinstance(parent_job, Job)
        assert isinstance(child_job, Job)
        assert isinstance(weight, (CalfWeight, TadpoleWeight))
        assert child_job.get_parent_job_to_weight()[parent_job] == weight
        if isinstance(weight, CalfWeight):
            assert isinstance(parent_job, ThinJob) == isinstance(child_job, ThinJob)
            assert parent_job.dimensions == child_job.dimensions
        else:
            assert isinstance(weight, TadpoleWeight)
            assert isinstance(parent_job, FatJob)
            assert parent_job.dimensions == child_job.dimensions + 1
        try:
            existing_edge_info = self.graph.edges[(parent_job, child_job)]
        except KeyError:
            self.graph.add_edge(parent_job, child_job, weight=weight)
        else:
            existing_weight = existing_edge_info['weight']
            assert weight == existing_weight


    def add_jobs(self, jobs: Union[Job, Iterable[Job]]) -> None:
        jobs: set = {jobs} if isinstance(jobs, Job) else set(jobs)
        jobs_to_process = set(jobs)
        jobs_processed = set()
        while jobs_to_process:
            job: Job = jobs_to_process.pop()
            self.graph.add_node(job)
            for parent_job, weight in job.get_parent_job_to_weight().items():
                self._add_edge(parent_job, job, weight)
                if parent_job not in jobs_processed:
                    jobs_to_process.add(parent_job)
            jobs_processed.add(job)

    def get_child_jobs(self, jobs: Union[Job, Iterable[Job]], *,
                       include_self: bool = False) -> Set[Job]:
        jobs: set = {jobs} if isinstance(jobs, Job) else set(jobs)
        child_jobs = set(itertools.chain.from_iterable(self.graph[job] for job in jobs))
        return (child_jobs | jobs) if include_self else child_jobs

    # def get_child_job_to_weight(self, job: Job) -> Dict[Job, Weight]:
        # child_jobs = {}
        # for child, d in self.graph.reverse()[job].items():
            # child_jobs[child] = d['weight']
        # return child_jobs


    def iterate_predecessor_jobs(self, jobs: Union[Job, Iterable[Job]], *,
                                 include_self: bool = False) -> Iterator[Job]:
        return self._iterate(jobs,
                             lambda job: tuple(job.get_parent_job_to_weight()),
                             include_self=include_self)


    def iterate_successor_jobs(self, jobs: Union[Job, Iterable[Job]], *,
                               include_self: bool = False) -> Iterator[Job]:
        return self._iterate(jobs,
                             lambda job: tuple(self.graph[job]),
                             include_self=include_self)


    def _iterate(self, jobs: Union[Job, Iterable[Job]], step: Callable[[Job], Iterable[Job]], *,
                 include_self: bool = False):
        jobs: set = {jobs} if isinstance(jobs, Job) else set(jobs)
        if include_self:
            yield from jobs
        all_jobs = set(jobs)
        while jobs:
            jobs = set(itertools.chain.from_iterable(map(step, jobs))) - all_jobs
            all_jobs |= jobs
            yield from jobs






