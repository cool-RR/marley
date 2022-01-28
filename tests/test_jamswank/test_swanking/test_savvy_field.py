# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

import tempfile
import json
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


Splat = MySwankDatabase = None

def setup_module():
    global Splat, MySwankDatabase

    class Splat(Swank):
        blat = SavvyField()

    class MySwankDatabase(SwankDatabase):
        pass

    return

class Cuteness:
    def __init__(self, *args):
        assert args == ('hi', 'there')

    def to_savvy_content(self):
        return ('hi', 'there')

    def __eq__(self, other):
        return isinstance(other, Cuteness)



pairs = (
    (
        {7: 'hi'},
        {'builtins.dict': [[{'builtins.tuple': [[7, 'hi']]}]]},
    ),
    (
        [1, 2, (3, 4)],
        [1, 2, {'builtins.tuple': [[3, 4]]}],
    ),
    (
        {1, 2, frozenset((3, 4))},
        {'builtins.set': [[{'builtins.frozenset': [[3, 4]]}, 1, 2]]},
    ),
    (
        [Cuteness('hi', 'there')],
        [{'tests.test_jamswank.test_swanking.test_savvy_field.Cuteness': ['hi', 'there']}],
    ),
    (
        [[[[5]]]],
        [[[[5]]]],
    ),
    (
        ((((5),),),),
        {'builtins.tuple': [[{'builtins.tuple': [[{'builtins.tuple': [[5]]}]]}]]},
    ),
)

def test_savvy_field():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)
        my_swank_database = MySwankDatabase(jam_file_database)

        splat = Splat(swank_database=my_swank_database)
        savvy_field = splat._Swank__fields['blat']

        for thing, savvy in pairs:
            assert savvy_field.to_jam(thing, my_swank_database) == savvy
            assert savvy_field.from_jam(savvy, my_swank_database, swank_cache={}) == thing
            savvy_after_json = json.loads(json.dumps(savvy))
            assert savvy == savvy_after_json
            assert repr(savvy) == repr(savvy_after_json)