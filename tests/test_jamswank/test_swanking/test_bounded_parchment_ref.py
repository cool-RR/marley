# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import tempfile
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
    SavvyField, JamId, utils,
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

def test_simple():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)
        my_swank_database: SwankDatabase = MySwankDatabase(jam_file_database)

        chuff = Chuff(swank_database=my_swank_database, name='puff_0')
        chuff.loots.start_index = 6
        chuff.loots.end_index = 10
        chuff.loots[0] = Loot(swank_database=my_swank_database, name='loot_0')
        chuff.loots[1] = Loot(swank_database=my_swank_database, name='loot_1')
        chuff.loots[2] = Loot(swank_database=my_swank_database, name='loot_2')
        chuff.loots.cached_length = None
        assert len(chuff.loots) == 4
        chuff.save()
        assert len(chuff.loots) == 4
        loot_jam_parchment: jamming.JamParchment = chuff.loots.jam_parchment
        assert len(loot_jam_parchment) == 9
        assert loot_jam_parchment[3].read_jam() is None
        assert loot_jam_parchment[6].read_jam() == {'name.simple': 'loot_0'}

        loot_3 = Loot(swank_database=my_swank_database, name='loot_3')
        with pytest.raises(marley.jamswank.swanking.parchment_field.CantExtendBoundedParchmentRef):
            chuff.loots.append(loot_3)
        chuff.loots[3] = loot_3
        chuff.save()
        assert chuff.loots[3].name == 'loot_3'
        assert loot_jam_parchment[9].read_jam() == {'name.simple': 'loot_3'}