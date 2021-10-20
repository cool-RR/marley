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
    SavvyField
)


Cat = Hat = MySwankDatabase = None

def setup_module():
    global Cat, Hat, MySwankDatabase

    class Cat(Swank):
        name = SimpleField()

    class Hat(Swank):
        stuff = SavvyField()

    class MySwankDatabase(SwankDatabase):
        pass

    return

@pytest.mark.parametrize('save_first', (False, True))
def test_swanks_in_savvy_field(save_first: bool):
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)
        my_swank_database = MySwankDatabase(jam_file_database)

        hat = Hat(swank_database=my_swank_database)

        furry_cat = Cat(swank_database=my_swank_database, name='furry')
        skinny_cat = Cat(swank_database=my_swank_database, name='skinny')
        fat_cat = Cat(swank_database=my_swank_database, name='fat')
        cats = (furry_cat, skinny_cat, fat_cat)
        if save_first:
            for cat in cats:
                cat.save()

        hat.stuff = {
            'furry': furry_cat,
            'skinny': skinny_cat,
            'fat': fat_cat,
        }
        hat.save()

        hat_reloaded = hat.reload()
        assert hat_reloaded.stuff == {
            'furry': furry_cat,
            'skinny': skinny_cat,
            'fat': fat_cat,
        }

