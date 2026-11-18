# -*- coding: utf-8 -*-
"""Headless screenshot capture / smoke test of the plane game.

Runs the real pygame loop with the dummy SDL video driver and a scripted
"hand" (no camera needed): the scripted source holds ``X11111`` to start the
round, then a representative battlefield is staged (enemies, a side-enemy
countdown and the invincibility ring) and the player is finally crashed into an
enemy to capture the game over screen.  Screenshots are written to ``docs/``.

Usage::

    .venv\\Scripts\\python.exe capture_game.py
"""

from __future__ import absolute_import

import os
import sys
import tempfile
from types import SimpleNamespace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

DOCS_DIR = os.path.join(ROOT, "docs")

import pygame  # noqa: E402

from game_core import (ENEMY_H, ENEMY_W, SIDE_ENEMY_SPEED, Bullet, Enemy,
                       HighScoreStore, PlaneGame, Popup, SideWarning)  # noqa: E402
from plane_game import run  # noqa: E402

#: A fully open hand: the pose that starts a round.
START_CODE = "P11111"

STATE = {
    "shots": [],
    "stage_at": None,
    "shot_at": None,
    "fail_at": None,
    "over_frames": 0,
}


class ScriptedSource(object):
    """Stands in for the camera worker: returns a fixed hand code."""

    def __init__(self):
        self.code = None

    def set(self, code):
        self.code = code

    def latest(self):
        return None, SimpleNamespace(code=self.code,
                                     direction="pause",
                                     start_pose=False)


def save_shot(name):
    screen = pygame.display.get_surface()
    os.makedirs(DOCS_DIR, exist_ok=True)
    pygame.image.save(screen, os.path.join(DOCS_DIR, name))
    STATE["shots"].append(name)
    print("[capture] saved docs/{}".format(name))


def stage_battlefield(game):
    """Stage a readable battle scene for the documentation screenshot."""
    game.score = 17
    game.elapsed = 26.0
    game.survived = 26.0
    game.difficulty.at(26.0)
    game.player.x, game.player.y = 200.0, 500.0
    game.invincible_timer = 2.2
    game.time = 0.0        # deterministic blink phase: the plane is visible
    game.enemies = [
        Enemy("top", 60.0, 140.0, vy=165.0),
        Enemy("top", 330.0, 210.0, vy=190.0),
        Enemy("side", 300.0, 430.0, vx=-SIDE_ENEMY_SPEED),
    ]
    game.warnings = [SideWarning("R", 610.0, 2.2)]
    game.bullets = [Bullet(221.0, 300.0), Bullet(221.0, 380.0)]
    game.popups = [Popup(358.0, 200.0, "+1", "score")]
    game.hit_flash = 0.0


def crash_into_enemy(game):
    """Put an enemy on the plane so the next update ends the round."""
    game.chances = 1
    game.invincible_timer = 0.0
    game.warnings = []
    player_x, player_y = game.player.center
    game.enemies = [Enemy("top", player_x - ENEMY_W / 2.0,
                          player_y - ENEMY_H / 2.0, vy=0.0)]


def make_hook(source):
    def hook(frame, game, current_source):
        if frame >= 1500:      # safety stop for the headless run
            return False

        if game.state == PlaneGame.READY:
            # Hold the start pose; the bar fills up and the round starts.
            source.set(START_CODE)
            if frame == 25:
                save_shot("screenshot_start.png")
            return None

        if game.state == PlaneGame.PLAYING:
            source.set(None)   # keep the plane still for the picture
            if STATE["stage_at"] is None and frame >= 100:
                stage_battlefield(game)
                STATE["stage_at"] = frame
                STATE["shot_at"] = frame + 2
            if STATE["shot_at"] is not None and frame >= STATE["shot_at"]:
                save_shot("screenshot_game.png")
                STATE["shot_at"] = None
                STATE["fail_at"] = frame + 4
            if STATE["fail_at"] is not None and frame >= STATE["fail_at"]:
                # The staged score is kept: _finish() only records it.
                crash_into_enemy(game)
                STATE["fail_at"] = None
            return None

        if game.state == PlaneGame.OVER:
            source.set(None)
            STATE["over_frames"] += 1
            # Wait for the hit flash to fade before taking the picture.
            if STATE["over_frames"] == 40:
                save_shot("screenshot_over.png")
                return False
        return None

    return hook


def main():
    pygame.init()
    source = ScriptedSource()

    # Keep the capture high score out of the real highscore.json.
    with tempfile.TemporaryDirectory() as folder:
        store = HighScoreStore(os.path.join(folder, "highscore.json"),
                               legacy_path=os.path.join(folder, "data.txt"))
        store.save({"best": 42, "last": 30, "games": 6})
        game = PlaneGame(seed=7, highscore=store)
        # The scene is created inside run(), after the display mode is set.
        result = run(source, hook=make_hook(source), game=game,
                     show_camera=False, fps=60)

    print("[capture] final state={} score={} best={} chances={}".format(
        result.state, result.score, result.best_score, result.chances))
    print("[capture] shots:", ", ".join(STATE["shots"]))
    pygame.quit()

    if result.state != PlaneGame.OVER:
        print("[capture] WARNING: the game did not reach the game over screen")
        return 1
    if len(STATE["shots"]) != 3:
        print("[capture] WARNING: expected 3 screenshots")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
