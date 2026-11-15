# -*- coding: utf-8 -*-
"""Run the real game loop off-screen and save a couple of screenshots.

This is a development helper used to verify the environment without keeping a
window open (and without a webcam).  It does not modify plane_cnn.py: the
game is executed through ``runpy`` with a dummy SDL video driver, the Play
button is "clicked" programmatically, and ``pygame.display.update`` is wrapped
to save the 480x800 framebuffer after a few frames.
"""

from __future__ import absolute_import

import os
import runpy
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

_real_update = pygame.display.update
_frames = {"count": 0}
_state = {"click": False}


def _update(*args, **kwargs):
    _frames["count"] += 1
    frame = _frames["count"]
    surface = pygame.display.get_surface()
    if surface is not None and frame == 25:
        pygame.image.save(surface, os.path.join(ROOT, "screenshot_start.png"))
        print("[capture] frame {} -> screenshot_start.png".format(frame))
        _state["click"] = True
    if surface is not None and frame == 300:
        pygame.image.save(surface, os.path.join(ROOT, "screenshot_game.png"))
        print("[capture] frame {} -> screenshot_game.png".format(frame))
    if frame >= 340:
        print("[capture] stopping after {} frames".format(frame))
        raise SystemExit(0)
    return _real_update(*args, **kwargs)


pygame.display.update = _update
# Pretend the mouse is clicking the centered Play button after frame 25.
pygame.mouse.get_pressed = lambda *a, **k: (1, 0, 0) if _state["click"] else (0, 0, 0)
pygame.mouse.get_pos = lambda *a, **k: (240, 400)

try:
    runpy.run_path(os.path.join(ROOT, "plane_cnn.py"), run_name="__main__")
except SystemExit:
    pass
finally:
    pygame.quit()
print("[capture] done")
