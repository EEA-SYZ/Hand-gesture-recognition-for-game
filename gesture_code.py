# -*- coding: utf-8 -*-
"""Gesture code helpers shared by the recogniser and the game logic.

A single hand is written as six characters::

    <P|N><thumb><index><middle><ring><pinky>

* ``P`` / ``N`` is the thumb orientation.  This project uses the same
  convention as the **right hand** of the two-hand rhythm game
  (``Hand-gesture-ps-game``): for a hand shown in the camera preview, ``P``
  means "the thumb points left" and ``N`` means "the thumb points right".
* The five digits are 1 (extended) / 0 (folded) for the thumb, index, middle,
  ring and pinky finger.

The plane game needs these poses, exactly as specified in the task:

======  ==================================================================
pose    meaning
======  ==================================================================
Xx1111  ``up``   - the four long fingers are extended (thumb free: ``x``)
X00000  ``down`` - a fist (thumb folded)
P10000  ``left`` - only the thumb is out and it points left
N10000  ``right``- only the thumb is out and it points right
X11111  start / restart - the whole hand is open, thumb included, held for
        0.7 s (a ``x1111`` pose with the thumb folded only moves the plane)
other   ``pause``- the plane keeps its current position
======  ==================================================================

``X`` therefore stands for ``P`` or ``N`` and ``x`` for ``1`` or ``0``.
Every other hand shape (a point, a peace sign, ...) is a valid *reading* but
does not move the plane.

This module has no third-party imports so the game logic can use it in tests
without a camera, MediaPipe or pygame.
"""

from __future__ import absolute_import

#: Allowed thumb tokens.
THUMBS = ("P", "N")

#: Directions understood by the game logic (``pause`` = do not move).
DIRECTIONS = ("up", "down", "left", "right", "pause")

#: Four long fingers extended: the ``1111`` part of ``Xx1111``.
LONG_FINGERS_UP = "1111"

#: The pose that starts (and restarts) a round: ``X11111``.
START_BITS = "11111"


def hand_code(thumb, bits):
    """Combine a thumb token and five finger bits into one 6 character code."""
    thumb = str(thumb).upper()
    bits = str(bits)
    if thumb not in THUMBS:
        raise ValueError("thumb must be 'P' or 'N': {!r}".format(thumb))
    if len(bits) != 5 or any(ch not in "01" for ch in bits):
        raise ValueError("bits must be 5 characters of 0/1: {!r}".format(bits))
    return thumb + bits


def parse_hand(code6):
    """Split a 6 character hand code into ``(thumb, bits)``."""
    if not isinstance(code6, str) or len(code6) != 6:
        raise ValueError("hand code must be 6 characters: {!r}".format(code6))
    thumb, bits = code6[0], code6[1:]
    hand_code(thumb, bits)  # validation only
    return thumb, bits


def is_hand_code(code6):
    """True when ``code6`` is a syntactically valid 6 character hand code."""
    try:
        parse_hand(code6)
    except (TypeError, ValueError):
        return False
    return True


def direction_of(code6):
    """Map a hand code to the plane direction (see the module docstring)."""
    if not is_hand_code(code6):
        return "pause"
    thumb, bits = parse_hand(code6)
    long_fingers = bits[1:]
    if long_fingers == LONG_FINGERS_UP:
        # Xx1111: four long fingers extended -> up.
        return "up"
    if bits == "00000":
        # X00000: fist -> down.
        return "down"
    if bits == "10000":
        # P10000 / N10000: only the thumb is out -> left / right.
        return "left" if thumb == "P" else "right"
    return "pause"


def is_start_pose(code6):
    """True for the ``X11111`` pose that starts (and restarts) a round.

    The whole hand has to be open, thumb included: ``P11111`` or ``N11111``.
    A ``Xx1111`` pose with the thumb folded only moves the plane up; a fist,
    a thumb up or no hand at all is not a start pose.  ``None`` and malformed
    codes are not start poses either.
    """
    if not is_hand_code(code6):
        return False
    _thumb, bits = parse_hand(code6)
    return bits == START_BITS


def format_code(code6):
    """Pretty print a hand code for debug output (``"--"`` when unknown)."""
    return code6 if is_hand_code(code6) else "--"
