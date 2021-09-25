# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import tempfile
import pathlib
import contextlib
import datetime as datetime_module

import pytest
import more_itertools

import marley
from marley.jamswank import jamming, JamFileDatabase


def test_basic():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)
        assert len(jam_file_database) == 0
        assert tuple(jam_file_database) == ()

        jam_kind = jam_file_database['my_kind']
        assert isinstance(jam_kind, jamming.JamKind)
        assert len(jam_kind) == 0

        jam_id = jamming.JamId.create()
        jam_parchment = jam_kind[jam_id]
        assert len(jam_parchment) == 0

        jam_item = jam_parchment[0]
        with pytest.raises(IndexError):
            jam_item.read_jam()
        with pytest.raises(IndexError):
            jam_item.read_text()

        jam = {'foo': 'yipee kye yay!', 'x': 5}
        jam_item.write_jam(jam)

        jam_after_reading = jam_item.read_jam()
        assert jam_after_reading == jam
        assert len(jam_parchment) == 1

        text = jam_item.read_text()
        assert isinstance(text, str)
        assert len(text) == jam_id.block_size

        text_again = jam_item.read_text()
        assert text == text_again


        other_jam_kind = jam_file_database['other_kind']
        assert isinstance(other_jam_kind, jamming.JamKind)
        assert len(other_jam_kind) == 0

        other_jam_id = jamming.JamId.create()
        other_jam_parchment = other_jam_kind[other_jam_id]
        assert len(other_jam_parchment) == 0

        other_jam = {'bar': 'baz'}
        other_jam_item = other_jam_parchment[0]
        other_jam_item.write_jam(other_jam)
        assert other_jam_item.read_jam() == {'bar': 'baz'}


def test_lots():
    with marley.utils.create_temp_folder() as temp_folder:
        jam_file_database = JamFileDatabase(temp_folder)
        jam_kind = jam_file_database['my_kind']
        jam_ids = [jamming.JamId.create() for _ in range(10)]
        for i, jam_id in enumerate(jam_ids):
            n_items = (2 ** i)
            jam_parchment = jam_kind[jam_id]
            jams = [{'yoohoo': j, 'whatever': str(j),
                     'datetime': datetime_module.datetime.now(),} for j in range(n_items)]
            for chunk in more_itertools.chunked(jams, 100):
                jam_parchment.extend_jams(chunk)
            assert len(jam_parchment) == n_items
            assert jam_parchment[0].read_jam()['yoohoo'] == 0
            assert jam_parchment[len(jam_parchment) - 1].read_jam()['yoohoo'] == n_items - 1

        penultimate_jam_parchment = jam_kind[jam_ids[-1]]
        jam = penultimate_jam_parchment[200].read_jam()
        assert jam['yoohoo'] == 200
        loaded_datetime = datetime_module.datetime.fromisoformat(jam['datetime'])
        assert type(loaded_datetime) is datetime_module.datetime



