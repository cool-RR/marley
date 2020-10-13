# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

import json
import math
import sys
import time
import itertools
import numbers
import collections.abc
import urllib.parse
from typing import Union, Optional, Iterator, Iterable, TypeVar, Callable, Tuple, TypeVar
import random

from browser import document, html, ajax, timer, window


CELL_SIZE = window.CELL_SIZE
HALF_CELL_SIZE = window.HALF_CELL_SIZE
BOARD_WIDTH = window.BOARD_WIDTH
BOARD_HEIGHT = window.BOARD_HEIGHT

def clamp(number, /, minimum, maximum):
    assert minimum <= maximum
    if number < minimum:
        return minimum
    elif number > maximum:
        return maximum
    else:
        return number



class Favorites:
    timepoint_text = property(lambda self: document.select('#timepoint-text')[0])
    timepoint_slider = property(lambda self: document.select('#timepoint-slider')[0])
    speed_slider = property(lambda self: document.select('#speed-slider')[0])
    play_pause_button = property(lambda self: document.select('button#play-pause')[0])
    update_button = property(lambda self: document.select('button#update-button')[0])
    keyboard_help_button = property(lambda self: document.select('button#keyboard-help-button')[0])
    play_pause_button_img = property(lambda self: document.select('button#play-pause img')[0])
    game_selector = property(lambda self: document.select('#game-selector')[0])
    loading_text = property(lambda self: document.select('#loading')[0])

favorites = Favorites()


_directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
_directions_names = ('up', 'right', 'down', 'left')

def update_button_handler(event=None) -> None:
    update_game_names()
    timeline.transition_fetcher.update()


def keyboard_help_button_handler(event=None) -> None:
    window.alert(keyboard_help_text)


keyboard_help_text = '''\
P  Play / pause

G  Go to start of simulation
H  Go 5 states back
J  Go 1 state back

K  Go 1 state forward
L Go 5 states forward
;  Go to end of simulation

I  Hold to slow down playback
O  Hold to speed up playback

?  Show this help screen
'''

Transition = TypeVar('Transition', bound=dict)

class Timeline(collections.abc.MutableSequence):

    def __init__(self) -> None:
        self._transitions = []
        self._needle = 0
        self._anchor_time = time.time()
        self._anchor_needle = self.needle
        self._anchor_speed = 3
        self._anchor_target = None
        self._anchor_play_after_target = None
        self.transition_fetcher = TransitionFetcher(self)
        self._game_name = None
        self._timer_id = None
        self.is_playing = False
        self.is_strong_playing = False
        self.speed = float(favorites.speed_slider.value)
        favorites.timepoint_slider.bind('input', self._on_timepoint_slider_change)
        favorites.speed_slider.bind('input', self._on_speed_slider_change)
        favorites.play_pause_button.bind('click', lambda event: self.toggle_playing())
        favorites.game_selector.bind(
            'change',
            lambda event: setattr(self, 'game_name',
                                  favorites.game_selector.selectedOptions[0].attrs['value'])
        )

    #######################################################################
    ### Defining sequence operations: #####################################
    #                                                                     #

    def __len__(self) -> int:
        return len(self._transitions)

    def __getitem__(self, i):
        return self._transitions[i]

    def __iter__(self) -> Iterator[Transition]:
        return iter(self._transitions)

    def __setitem__(self, i: Union[int, slice],
                    value: Union[Iterable[Transition], Transition]) -> None:
        raise NotImplementedError
        # self._transitions[i] = value

    def __delitem__(self, index: Union[int, slice]) -> None:
        del self._transitions[index]

    def insert(self, index: int, value: Transition) -> None:
        assert index == len(self._transitions) # Can only add items at the end
        self._transitions.insert(index, value)
        window.update_ui()

    def extend(self, values: Iterable[Transition]) -> None:
        for v in values:
            self.append(v)

    #                                                                     #
    ### Finished defining sequence operations. ############################
    #######################################################################


    @property
    def needle(self) -> float:
        return self._needle

    @needle.setter
    def needle(self, needle: float) -> None:
        self._needle = self._clamp_needle(needle)
        self._set_anchor()

    def _clamp_needle(self, needle):
        return clamp(round(needle), 0, max((len(self) - 1, 0)))


    def _on_timepoint_slider_change(self, event):
        self.needle = int(favorites.timepoint_slider.value)

    def _on_speed_slider_change(self, event):
        self.speed = float(favorites.speed_slider.value)
        self._set_anchor()

    def get_active(self) -> Transition:
        return self._transitions[math.ceil(self.needle)]

    def _set_anchor(self):
        self._anchor_time = time.time()
        self._anchor_needle = self.needle
        self._anchor_speed = self.speed
        self._anchor_target = False
        self._anchor_play_after_target = False


    def play(self, *, change_icon: bool = True) -> None:
        if not self.is_strong_playing:
            self._set_anchor()
            if self.needle == len(self) - 1:
                self.needle = 0
            self.is_playing = self.is_strong_playing = True
            if change_icon:
                favorites.play_pause_button_img.attrs['src'] = 'pause.png'


    def pause(self, *, change_icon: bool = True) -> None:
        self.is_playing = self.is_strong_playing = False
        if change_icon:
            favorites.play_pause_button_img.attrs['src'] = 'play.png'
        self.needle = round(self.needle)

    def toggle_playing(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()


    def skip(self, delta: numbers.Real) -> None:
        was_strong_playing = self.is_strong_playing

        target_needle = self._clamp_needle(self.needle + delta)

        distance_to_cover = target_needle - self.needle
        time_to_cover_distance = clamp(0.1 + ((abs(distance_to_cover) - 1) / 10),
                                       minimum=0.1, maximum=0.75)

        self._anchor_time = time.time()
        self._anchor_needle = self.needle
        self._anchor_speed = distance_to_cover / time_to_cover_distance
        self._anchor_target = target_needle
        self._anchor_play_after_target = was_strong_playing
        self.is_playing = True


    @property
    def game_name(self) -> str:
        return self._game_name

    @game_name.setter
    def game_name(self, new_game_name: str) -> None:
        self.pause()
        self._game_name = new_game_name
        self.clear()
        self._transitions.clear()
        self.needle = 0
        self.transition_fetcher.update()

    def change_speed(self, delta: numbers.Real) -> None:
        self.speed = clamp(self.speed + delta, 0.2, 8)
        window.update_ui()
        self._set_anchor()



class TransitionFetcher:
    delay = 20_000
    def __init__(self, timeline: Timeline) -> None:
        self.timeline = timeline
        self._timer_id = None
        self.game_name = None
        self.current_batch = 0

    def update(self):
        game_name = self.timeline.game_name
        if game_name != self.game_name:
            self.current_batch = 0
            self.game_name = game_name
        timer.clear_timeout(self._timer_id)
        self._do_next()

    def _do_next(self):
        def _complete(request):
            if request.status == 200:
                self.timeline.extend(json.loads(request.responseText))
                self.current_batch += 1
                self._timer_id = timer.set_timeout(self._do_next, 0)
                favorites.loading_text.style.display = 'none' # On first update we hide it.
            elif request.status == 404:
                self._timer_id = timer.set_timeout(self._do_next, self.delay)
                pass
            else:
                raise NotImplementedError

        url = f'games/{self.game_name}/{self.current_batch:06d}.json'
        cool_ajax(url, _complete)



timeline = window.timeline = Timeline()

key_bindings = {
    'p': timeline.toggle_playing,

    'g': lambda: timeline.skip(- float('inf')),
    'h': lambda: timeline.skip(-5),
    'j': lambda: timeline.skip(-1),
    'k': lambda: timeline.skip(1),
    'l': lambda: timeline.skip(5),
    ';': lambda: timeline.skip(len(timeline)), #float('inf')),

    'i': lambda: timeline.change_speed(-0.2),
    'o': lambda: timeline.change_speed(0.2),

    'u': update_button_handler,

    '?': keyboard_help_button_handler,
}

def keypress_handler(event):
    key = event.key.lower()
    try:
        function = key_bindings[key]
    except KeyError:
        pass
    else:
        function()

def add_parameters_to_url(url: str, parameters: dict) -> str:
    # Todo: This is a caveman implementation, replace with urllib.parse
    if '?' in url:
        raise NotImplementedError
    return f'{url}?{"&".join(f"{key}={value}" for key, value in parameters.items())}'



def cool_ajax(url: str, handler: Callable, method: str = 'GET', disable_cache: bool = True) -> None:
    request = ajax.ajax()
    if disable_cache:
        url = add_parameters_to_url(url, {'_': random.randint(0, 10**8)})
    request.open(method, url, True)
    request.bind('complete', handler)
    request.send()


def update_game_names(*, loop: bool = False, select_newest: bool = False) -> None:
    def _complete(request):
        game_names = json.loads(request.responseText)
        for game_name in game_names:
            if not document.select(f'#game-selector option[value="{game_name}"]'):
                option = html.OPTION(value=game_name)
                option.textContent = game_name
                favorites.game_selector <= option
        if select_newest:
            latest_game_name = get_latest_game_name()
            latest_game_name_option = document.select(
                                      f'#game-selector option[value="{get_latest_game_name()}"]')[0]
            latest_game_name_option.selected = True
            timeline.game_name = latest_game_name
        if loop:
            timer.set_timeout(lambda: update_game_names(loop=True), 20_000)
    cool_ajax('games', _complete)

def get_latest_game_name():
    return max(option.attrs['value'] for option in document.select(f'#game-selector option'))


def main():
    document.bind('keypress', keypress_handler)
    update_game_names(loop=True, select_newest=True)
    window._animate_lock = False
    favorites.update_button.bind('click', update_button_handler)
    favorites.keyboard_help_button.bind('click', keyboard_help_button_handler)

    timer.request_animation_frame(window.animate)



main()