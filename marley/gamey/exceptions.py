# Copyright 2020 Ram Rachum and collaborators.
# This program is distributed under the MIT license.

from __future__ import annotations

from typing import Optional

from .. import gamey


class GameyException(Exception):
    pass

class IllegalAction(GameyException):
    def __init__(self, action: Optional[gamey.Action] = None) -> None:
        self.action = action or "the given action"
        GameyException.__init__(self, f"Can't play {action} in this state.")


class GameOver(GameyException):
    '''You tried to go forward in a game that's already over.'''
    pass
