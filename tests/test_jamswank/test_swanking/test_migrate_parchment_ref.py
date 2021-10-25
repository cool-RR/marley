# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import tempfile
import itertools
import pathlib
import contextlib
import datetime as datetime_module
import enum

import pytest
import more_itertools

from marley import gamey
import marley
from marley.jamswank import jamming, JamFileDatabase
from marley.jamswank.swanking import (
    Swank, SwankDatabase, SimpleField, ParchmentField, SwankField,
    SavvyField, JamId, utils
)


Loot = Chuff = MySwankDatabase = None

def setup_module():
    global Loot, Chuff, MySwankDatabase

    class Loot(Swank):
        name = SimpleField()

    class Chuff(Swank):
        name = SimpleField()
        loots = ParchmentField()

    class MySwankDatabase(SwankDatabase):
        pass

    return


def test_migrate_to_new():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)
        my_swank_database: SwankDatabase = MySwankDatabase(jam_file_database)

        chuff = Chuff(swank_database=my_swank_database, name='puff_0')
        chuff.loots = (
            Loot(swank_database=my_swank_database, name='loot_0'),
            Loot(swank_database=my_swank_database, name='loot_1'),
            Loot(swank_database=my_swank_database, name='loot_2'),
        )
        chuff.save()
        first_loot_jam_parchment: jamming.JamParchment = chuff.loots.jam_parchment
        assert first_loot_jam_parchment.jam_id is not None

        loot_jam_kind = my_swank_database.jam_database[Loot]
        second_loot_jam_parchment = loot_jam_kind[JamId.create(block_size=Loot.default_block_size)]
        chuff.loots.migrate(second_loot_jam_parchment.jam_id)

        assert len(first_loot_jam_parchment) == 0
        assert not first_loot_jam_parchment._get_path().exists()
        assert len(second_loot_jam_parchment) == 3
        assert chuff.loots[0].name == 'loot_0'
        assert chuff.loots[-1].name == 'loot_2'


def test_migrate_after_existing():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)
        my_swank_database: SwankDatabase = MySwankDatabase(jam_file_database)

        first_chuff = Chuff(swank_database=my_swank_database, name='puff_0')
        first_chuff.loots = (
            Loot(swank_database=my_swank_database, name='loot_0'),
            Loot(swank_database=my_swank_database, name='loot_1'),
            Loot(swank_database=my_swank_database, name='loot_2'),
        )
        first_chuff.save()

        second_chuff = Chuff(swank_database=my_swank_database, name='puff_0')
        second_chuff.loots = (
            Loot(swank_database=my_swank_database, name='loot_3'),
            Loot(swank_database=my_swank_database, name='loot_4'),
            Loot(swank_database=my_swank_database, name='loot_5'),
        )
        second_chuff.save()

        first_loot_jam_parchment: jamming.JamParchment = first_chuff.loots.jam_parchment
        second_loot_jam_parchment: jamming.JamParchment = second_chuff.loots.jam_parchment

        first_chuff.loots.end_index = 3
        first_chuff.save()
        second_chuff.loots.migrate(first_chuff.loots.jam_id, start_index=3, end_index=6)
        second_chuff.save()

        first_chuff = first_chuff.reload()
        second_chuff = second_chuff.reload()

        assert len(first_chuff.loots) == len(second_chuff.loots)
        assert tuple(
            loot.name for loot in itertools.chain(first_chuff.loots, second_chuff.loots)) == tuple(
            f'loot_{i}' for i in range(6)
        )

        assert len(first_loot_jam_parchment) == 6
        assert not second_loot_jam_parchment._get_path().exists()
        assert len(second_loot_jam_parchment) == 0