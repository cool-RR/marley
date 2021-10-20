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

class WhateverEnum(enum.Enum):
    foo = 'foo'
    bar = 'bar'


Roof = Puff = Snipe = Zen = MySwankDatabase = None

def setup_module():
    global Roof, Puff, Snipe, Zen, MySwankDatabase

    class Roof(Swank):
        yee = SimpleField()
        fee = SimpleField()

    class Puff(Swank):
        foo = SimpleField()
        bar = SimpleField()
        roofs = ParchmentField()

    class Snipe(Swank):
        zux = SimpleField()
        left_roof = SwankField()
        right_roof = SwankField()

    class Zen(Swank):
        default_block_size = 10_000
        snipe_dict = SavvyField(lambda: {})
        snipe_parchment = ParchmentField()
        i = SimpleField()
        whatever = SavvyField()

    class MySwankDatabase(SwankDatabase):
        pass

    return

def test_parchment_ref():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)

        my_swank_database = MySwankDatabase(jam_file_database)

        roof = Roof(swank_database=my_swank_database, yee=7, fee='feefee')
        roof.save()
        assert roof.jam_index == 0

        roof_copy: Roof = my_swank_database.load_swank(Roof, roof.jam_id, roof.jam_index)
        assert roof == roof_copy
        assert roof_copy.yee == 7
        assert roof_copy.fee == 'feefee'

        another_roof = Roof(swank_database=my_swank_database, yee=9)
        assert another_roof.fee is None
        assert not another_roof.has_jam_id_and_index
        another_roof.save()
        assert another_roof.has_jam_id_and_index
        another_roof_copy: Roof = my_swank_database.load_swank(Roof, another_roof.jam_id,
                                                               another_roof.jam_index)
        assert another_roof_copy == another_roof
        assert another_roof_copy.yee == 9
        assert another_roof_copy.fee is None

        puff = Puff(swank_database=my_swank_database)
        assert len(puff.roofs) == 0
        assert not puff.roofs
        for i in (0, 1, 3, -1):
            with pytest.raises(IndexError):
                puff.roofs[i]

        puff.roofs.extend(Roof(swank_database=my_swank_database) for _ in range(100))
        assert len(puff.roofs) == 100
        assert puff.roofs
        for i in (0, 1, 3, -1):
            roof = puff.roofs[i]
            assert isinstance(roof, Roof)
            assert roof.yee is roof.fee is None
            if i != -1:
                assert roof.jam_index == i
            else:
                assert roof.jam_index == 99

        puff.roofs.save()
        assert puff.roofs
        puff.save()

        puff_copy: Puff = my_swank_database.load_swank(Puff, puff.jam_id, puff.jam_index)
        assert len(puff_copy.roofs) == 100
        assert puff_copy.roofs
        for i in (0, 1, 3, -1):
            roof = puff_copy.roofs[i]
            assert isinstance(roof, Roof)
            assert roof.yee is roof.fee is None
            if i != -1:
                assert roof.jam_index == i
            else:
                assert roof.jam_index == 99


def test_item_ref():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)

        my_swank_database = MySwankDatabase(jam_file_database)

        snipe = Snipe(swank_database=my_swank_database)
        assert snipe.zux is snipe.left_roof is snipe.right_roof is None
        snipe.save()

        assert snipe.zux is snipe.left_roof is snipe.right_roof is None

        snipe_copy = Snipe.load(my_swank_database, snipe.jam_id, snipe.jam_index)
        assert snipe_copy.zux is snipe_copy.left_roof is snipe_copy.right_roof is None

        lonely_roof = Roof(swank_database=my_swank_database, yee='lonely')
        lonely_roof.save()

        puff = Puff(swank_database=my_swank_database)
        puff.save()
        puff.roofs.extend(Roof(swank_database=my_swank_database, fee=i) for i in range(100))
        puff.roofs.save()

        snipe_copy.left_roof = lonely_roof
        snipe_copy.right_roof = puff.roofs[7]
        snipe_copy.save()

        snipe_extra_copy = Snipe.load(my_swank_database, snipe.jam_id, snipe.jam_index)

        assert snipe_extra_copy.left_roof == lonely_roof
        assert snipe_extra_copy.right_roof == puff.roofs[7]

        assert snipe_copy.right_roof.fee == 7

        snipes = []
        for left_roof, right_roof in gamey.utils.iterate_windowed_pairs(puff.roofs):
            snipe = Snipe(swank_database=my_swank_database, left_roof=left_roof,
                          right_roof=right_roof)
            snipe.save()
            snipes.append(snipe)

        snipe_14, snipe_15 = snipes[14:16]
        assert snipe_14.left_roof.fee == 14
        assert snipe_14.right_roof == snipe_15.left_roof



def test_savvy_field():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)

        my_swank_database = MySwankDatabase(jam_file_database)

        zen = Zen(swank_database=my_swank_database)
        zen.save()
        assert zen.snipe_dict == {}
        assert not zen.snipe_parchment


        lonely_snipe = Snipe(swank_database=my_swank_database, zux='lonely')
        assert lonely_snipe.zux == 'lonely'
        lonely_snipe.save()

        zen.snipe_dict['a'] = zen.snipe_dict['b'] = lonely_snipe
        zen.save()

        zen_copy = Zen.load(my_swank_database, zen.jam_id, zen.jam_index)
        assert zen_copy.snipe_dict == {'a': lonely_snipe, 'b': lonely_snipe}

        zen.snipe_parchment.extend(
            Snipe(swank_database=my_swank_database, zux=i) for i in range(100)
        )
        zen.snipe_parchment.save()
        zen.snipe_dict['b'] = zen.snipe_dict['ccc'] = zen.snipe_parchment[77]
        zen.save()

        zen_copy = Zen.load(my_swank_database, zen.jam_id, zen.jam_index)
        assert zen_copy.snipe_dict == {'a': lonely_snipe, 'b': zen.snipe_parchment[77],
                                       'ccc': zen.snipe_parchment[77],}

        assert zen.whatever is None
        zen.whatever = [
            ({'oy': zen.snipe_dict['ccc']}, {'oy': zen.snipe_dict['ccc']}, 7, 'asdf'),
            WhateverEnum.foo,
            {1, 2, 3},
            frozenset((1, 2, 3)),
            {('a',): ('a',)}
        ]
        zen.save()

        zen_copy = Zen.load(my_swank_database, zen.jam_id, zen.jam_index)
        assert zen_copy.whatever == [
            ({'oy': zen.snipe_dict['ccc']}, {'oy': zen.snipe_dict['ccc']}, 7, 'asdf'),
            WhateverEnum.foo,
            {1, 2, 3},
            frozenset((1, 2, 3)),
            {('a',): ('a',)}
        ]

