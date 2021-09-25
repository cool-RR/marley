# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import abc
from typing import (Iterable, Union, Optional, Tuple, Any, Iterator, Type,
                    Sequence, Callable, Hashable, Mapping, TypeVar, Dict)
import threading
import functools
import collections
import logging
import itertools
import random
import operator as operator_module
import uuid as uuid_module
import contextlib

import more_itertools

from .weighting import Weight, CalfWeight, TadpoleWeight
from .gaining import Gain, ThinGain, FatGain, GainArg, GainDyad
from .jobbing import Job, ThinJob, FatJob, JobSniffingJob
from .wedging import Wedge
from . import utils

import networkx as nx

logger = logging.getLogger(__name__)


class JobToGainDict(utils.CuteUserDict):
    def __setitem__(self, job: Job, gain: Gain) -> None:
        if gain:
            collections.UserDict.__setitem__(self, job, gain)
        else:
            try:
                del self[job]
            except KeyError:
                pass

class JobToItsGainDict(JobToGainDict):
    def __missing__(self, job: Job) -> Gain:
        return job.create_gain()

    def __setitem__(self, job: Job, gain_or_arg: Union[Gain, GainArg]) -> None:
        from .int_crowding import IntCrowd
        if isinstance(gain_or_arg, Gain):
            assert gain_or_arg.job == job
            gain = gain_or_arg
        else:
            assert isinstance(gain_or_arg, (IntCrowd, bool, type(None)))
            gain = job.create_gain(gain_or_arg)

        JobToGainDict.__setitem__(self, job, gain)



class JobToItsDirectiveGainDict(JobToItsGainDict):
    def __init__(self, antilles: Antilles, *args, **kwargs) -> None:
        self.antilles = antilles
        JobToItsGainDict.__init__(self, *args, **kwargs)


    def __setitem__(self, job: Job, gain: Gain) -> None:
        self.antilles.wedge.add_jobs(job)
        with self.antilles.lock, self.antilles._write_diff():
            if gain != self.get(job):
                JobToItsGainDict.__setitem__(self, job, gain)
                self.antilles._update_desired_and_available_gains_of_all_jobs()

    def __delitem__(self, job: Job) -> None:
        with self.antilles.lock, self.antilles._write_diff():
            gain = self.data.pop(job, None)
            if gain:
                self.antilles._update_desired_and_available_gains_of_all_jobs()

    def add_thin_jobs(self, thin_jobs: Union[ThinJob, Iterable[ThinJob]]) -> None:
        thin_jobs = (thin_jobs,) if isinstance(thin_jobs, ThinJob) else thin_jobs
        for thin_job in thin_jobs:
            self[thin_job] = True




class Antilles:
    def __init__(self, *, name: Optional[str] = None, keep_diffs: bool = True) -> None:
        self.name = name or uuid_module.uuid4().hex[:6]
        self.wedge = Wedge()
        self.lock = threading.RLock()

        self.keep_diffs = keep_diffs
        self.diffs = [] if keep_diffs else None

        # Gains that were run are called "finished":
        self.job_to_finished_gain = JobToItsGainDict()

        # Gains that the supervisor is working on are called "pending":
        self.job_to_pending_gain = JobToItsGainDict()

        # Gains that were finished in a previous run:
        self.job_to_preexisting_gain = JobToItsGainDict()

        # Gains that were sniffed, i.e. we checked for preexisting gain:
        self.job_to_sniffed_gain = JobToItsGainDict()

        # Gains that the supervisor was given to sniff:
        self.job_to_sniff_pending_gain = JobToItsGainDict()

        # Gains that we were explicitly asked by the user to run are called "directive":
        self.job_to_directive_gain = JobToItsDirectiveGainDict(self)

        # Gains that we want to be finished are called "desired", even if they're already finished:
        self.job_to_desired_gain = JobToItsGainDict()
        # A gain is desired either if it's directive, or if it's a dependency of a directive gain.

        # Gains that are desired, were not given to the supervisor, and are not blocked by any
        # dependencies are called "available":
        self.job_to_available_gain = JobToItsGainDict()

        self.job_to_child_job_to_desired_gain_from_it = collections.defaultdict(JobToGainDict)

        self.doable_available_gain_log = [] # todo remove?


    # def _update_desired_and_available_gain_from_directive_and_children(self, job: Job) -> None:
        # self.job_to_desired_gain[job] = utils.union((self.job_to_directive_gain[job],) + tuple(
                                     # self.job_to_child_job_to_desired_gain_from_it[job].values()))
        # self._update_available_gains_for_jobs(job)


    def _update_desired_and_available_gains_of_all_jobs(self) -> None:
        # Inefficient implementation.
        with self.lock:
            job_to_desired_gain = JobToItsGainDict()
            job_to_child_job_to_desired_gain_from_it = collections.defaultdict(JobToGainDict)

            all_jobs = set(self.wedge.iterate_predecessor_jobs(self.job_to_directive_gain,
                                                               include_self=True))
            dirty_jobs = set(all_jobs)
            while dirty_jobs:
                job = random.choice(tuple(dirty_jobs))
                dirty_jobs.remove(job)
                old_guess_for_desired_gain = job_to_desired_gain[job]
                current_guess_for_desired_gain = (
                    utils.union((self.job_to_directive_gain[job],) + tuple(
                    job_to_child_job_to_desired_gain_from_it[job].values()))
                ).get_gain_desired_by_self()
                if current_guess_for_desired_gain != old_guess_for_desired_gain:
                    job_to_desired_gain[job] = current_guess_for_desired_gain
                    sniffed_not_preexisting_part_of_current_guess_for_desired_gain = (
                        current_guess_for_desired_gain & self.job_to_sniffed_gain[job] -
                        self.job_to_preexisting_gain[job]
                    )
                    dirty_jobs.update(job.get_parent_job_to_weight())

                    for parent_job, weight in job.get_parent_job_to_weight().items():
                        weight: Weight
                        desired_gain_from_child = weight.get_desired_parent_gain(
                            sniffed_not_preexisting_part_of_current_guess_for_desired_gain,
                            parent_job
                        )
                        job_to_child_job_to_desired_gain_from_it[parent_job][job] = \
                                                                             desired_gain_from_child

            self.job_to_desired_gain = job_to_desired_gain
            self.job_to_child_job_to_desired_gain_from_it = job_to_child_job_to_desired_gain_from_it

            self._update_available_gains(all_jobs)

            for job in (set(self.job_to_available_gain) - set(all_jobs)):
                if not isinstance(job, JobSniffingJob):
                    del self.job_to_available_gain[job]




    def report_finished_gain_dyads(self, gain_dyads: Iterable[GainDyad]) -> None:
        with self.lock, self._write_diff():
            jobs_that_have_new_finished_gains = set()
            need_to_update_desired_gains_of_all_jobs = False
            for gain_dyad in gain_dyads:
                gain_dyad: GainDyad
                # print(gain_dyad)
                requested_gain = gain_dyad.requested_gain
                returned_gain = gain_dyad.returned_gain
                assert requested_gain in self.job_to_pending_gain[requested_gain.job]
                assert requested_gain not in self.job_to_finished_gain[requested_gain.job]
                if isinstance(requested_gain.job, JobSniffingJob):
                    need_to_update_desired_gains_of_all_jobs = True
                    job_sniffing_job = requested_gain.job
                    target_gain = job_sniffing_job.target_gain
                    assert target_gain in self.job_to_sniff_pending_gain[target_gain.job]
                    assert target_gain not in self.job_to_sniffed_gain[target_gain.job]
                    self.job_to_sniff_pending_gain[target_gain.job] -= target_gain
                    self.job_to_sniffed_gain[target_gain.job] |= target_gain
                    self.job_to_pending_gain[requested_gain.job] -= requested_gain
                    self.job_to_finished_gain[requested_gain.job] |= requested_gain
                    if returned_gain:
                        assert returned_gain.job == target_gain.job
                        self.job_to_finished_gain[returned_gain.job] |= returned_gain
                        self.job_to_preexisting_gain[returned_gain.job] |= returned_gain
                        jobs_that_have_new_finished_gains.add(returned_gain.job)
                else:
                    finished_gain = (gain_dyad.requested_gain if gain_dyad.returned_gain is None
                                     else gain_dyad.returned_gain)
                    self.job_to_pending_gain[requested_gain.job] -= finished_gain
                    self.job_to_finished_gain[requested_gain.job] |= finished_gain
                    jobs_that_have_new_finished_gains.add(requested_gain.job)


            if need_to_update_desired_gains_of_all_jobs:
                self._update_desired_and_available_gains_of_all_jobs()
            else:
                directly_affected_jobs = self.wedge.get_child_jobs(
                    jobs_that_have_new_finished_gains,
                    include_self=True
                )
                self._update_available_gains(directly_affected_jobs)

            for job in (set(self.job_to_available_gain) | set(self.job_to_finished_gain)):
                assert not self.job_to_available_gain[job] & self.job_to_finished_gain[job]

    def _update_available_gains(self, jobs: Union[Job, Iterable[Job]]) -> None:
        # todo: The more efficient design would involve getting the new gains for the parents, so
        # we could calculate only the delta instead of the entire thing. But we can hold off on that
        # for now.
        with self.lock:
            jobs: set = {jobs} if isinstance(jobs, Job) else set(jobs)
            for job in jobs:
                job: Job
                desired_gain: Gain = self.job_to_desired_gain[job]
                desired_unsniffed_not_sniff_pending_gain = (
                    desired_gain - self.job_to_sniffed_gain[job] -
                    self.job_to_sniff_pending_gain[job]
                )
                if desired_unsniffed_not_sniff_pending_gain:
                    self.job_to_available_gain[
                                    JobSniffingJob(desired_unsniffed_not_sniff_pending_gain)] = True
                desired_sniffed_not_preexisting_gain = (
                    desired_gain & self.job_to_sniffed_gain[job] -
                    self.job_to_preexisting_gain[job]
                )
                unblocked_gain_components = [
                    desired_sniffed_not_preexisting_gain.
                                                    get_gain_unblocked_by_finished_gain_of_same_job(
                                                            self.job_to_finished_gain[job]
                                                        )
                ]
                for parent_job, weight in job.get_parent_job_to_weight().items():
                    unblocked_gain_components.append(
                        desired_sniffed_not_preexisting_gain.
                                                         get_gain_unblocked_by_finished_parent_gain(
                            self.job_to_finished_gain[parent_job],
                            weight
                        )
                    )

                self.job_to_available_gain[job] = (utils.intersection(unblocked_gain_components) -
                                                   self.job_to_finished_gain[job] -
                                                   self.job_to_pending_gain[job])

    def iterate_doable_available_gains_and_mark_pending(self, *,
                                                        sniff_only: bool = False) -> Iterator[Gain]:
        while True:
            with self.lock:
                if sniff_only:
                    job_pool = tuple(job for job in self.job_to_available_gain
                                     if isinstance(job, JobSniffingJob))
                else:
                    job_pool = tuple(self.job_to_available_gain)
                if not job_pool:
                    break
                job = random.choice(job_pool)
                available_gain = self.job_to_available_gain[job]
                if not available_gain:
                    continue
                doable_available_gain = available_gain.get_doable()
                assert doable_available_gain not in self.doable_available_gain_log
                with self._write_diff():
                    self.doable_available_gain_log.append(doable_available_gain)
                    self.job_to_available_gain[job] -= doable_available_gain
                    self.job_to_pending_gain[job] |= doable_available_gain
                    if isinstance(doable_available_gain.job, JobSniffingJob):
                        target_gain = doable_available_gain.job.target_gain
                        self.job_to_sniff_pending_gain[target_gain.job] |= target_gain
            yield doable_available_gain


    @contextlib.contextmanager
    def _write_diff(self) -> AntillesDiffRecorder:
        if not self.keep_diffs:
            yield
        else:
            with AntillesDiffRecorder(self) as antilles_change_recorder:
                yield
            if antilles_change_recorder.antilles_diff:
                self.diffs.append(antilles_change_recorder.antilles_diff)
                logger.debug(f'Antilles diff:\n{antilles_change_recorder.antilles_diff.show()}')

    def _snapshot(self) -> AntillesSnapshot:
        return AntillesSnapshot.create_from_antilles(self)


    def diffs_as_dict(self) -> dict:
        assert self.keep_diffs
        return tuple(antilles_diff.show() for antilles_diff in self.diffs)

    def show_diffs(self) -> str:
        assert self.keep_diffs
        return '\n'.join(antilles_diff.show() for antilles_diff in self.diffs)


from .antilles_diffing import AntillesDiffRecorder, AntillesSnapshot