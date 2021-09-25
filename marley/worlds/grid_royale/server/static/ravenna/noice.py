# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import json
import math
import sys
import re
import time
import itertools
import numbers
import collections.abc
import urllib.parse
from typing import Union, Optional, Iterator, Iterable, TypeVar, Callable, Tuple, TypeVar, Sequence
import random

from browser import document, html, ajax, timer, window

from .utils import cool_ajax, add_parameters_to_url, sliced


jam_id_pattern = re.compile(
    r'^([0-9]{8})-([0-9]{6})-([0-9]{6})-([0-9a-z]{30})\.([0-9]{12})$'
)
jam_kind_pattern = re.compile('[a-z_]+')


class JellyDatabase:
    def __init__(self, url: str) -> None:
        assert url.endswith('/')
        self.url = url


    def _make_request(self, path: str, callback: Callable, *, binary: bool = False,
                      disable_cache: bool = True) -> None:
        url = f'{self.url}{path}'
        if disable_cache:
            url = add_parameters_to_url(url, {'_': random.randint(0, 10**8)})
        def inner(ajax):
            try:
                callback(ajax)
            except:
                import sys
                sys.excepthook(*sys.exc_info())
                raise

        ajax.get(url, mode=('binary' if binary else 'text'), oncomplete=inner)


    def get_jam_parchment_names(self, jam_kind_name: str,
                                callback: Callable[[Sequence[str]], None]) -> None:
        assert jam_kind_pattern.fullmatch(jam_kind_name)
        def inner(request: ajax.ajax) -> None:
            jam_parchment_names = json.loads(request.read())
            callback(jam_parchment_names)

        self._make_request(jam_kind_name, inner)

    def read_texts(self, jam_kind_name: str, jam_id_name: str, start: int, end: Optional[int], *,
                   callback: Callable) -> bytes:
        assert jam_kind_pattern.fullmatch(jam_kind_name)
        assert jam_id_pattern.fullmatch(jam_id_name)
        block_size = int(jam_id_pattern.fullmatch(jam_id_name).groups()[-1])
        smart_end = end if (end is not None) else float('inf')
        assert 0 <= start <= smart_end
        def inner(request: ajax.ajax) -> None:
            conjoined_texts = request.read()
            assert isinstance(conjoined_texts, bytes)
            assert len(conjoined_texts) % block_size == 0
            texts = tuple(map(bytes.decode, sliced(conjoined_texts, block_size)))
            assert 0 <= len(texts) <= (smart_end - start)
            callback(texts)

        self._make_request(f'{jam_kind_name}/{jam_id_name}/{start}..{end}', inner, binary=True)

    def read_jams(self, jam_kind_name: str, jam_id_name: str, start: int, end: int, *,
                  callback: Callable) -> bytes:
        self.read_texts(jam_kind_name, jam_id_name, start, end,
                        callback=(lambda texts: callback(tuple(map(json.loads, texts)))))

    # def _get_jam_kinds(self) -> Sequence[JamKind]:
        # response = self._make_request('')
        # result = response.json()
        # assert isinstance(result, list)
        # assert set(map(type, result)) == {str}
        # return result

    # def _get_jam_kind_length(self, jam_kind_name: str) -> int:
        # return len(self._get_jam_parchments(jam_kind_name))

    # def _get_jam_parchment_length(self, jam_kind_name: str, jam_id: Union[str, JamId]) -> int:
        # (jam_id, jam_id_name) = JamId.parse(jam_id)
        # assert re.fullmatch('^[a-z0-9_]+$', jam_kind_name)
        # response = self._make_request(f'{jam_kind_name}/{jam_id_name}/length')
        # result = response.text
        # assert re.fullmatch('^[0-9]+$', result)
        # return int(result)





