# -*- coding: utf-8 -*-
"""Unit tests for the plane game logic (no pygame, no camera, no display).

Covers the difficulty model, the 0.7 s hold-to-start rule, chances and
invincibility, the circular hit boxes, the side-enemy countdown and the high
score persistence.

Run with::

    .venv\\Scripts\\python.exe test_game_logic.py
"""

from __future__ import absolute_import

import os
import tempfile

from game_core import (ENEMY_H, ENEMY_HIT_RADIUS, ENEMY_W, HOLD_SECONDS,
                       INVINCIBLE_SECONDS, MAX_CHANCES, PLAYER_HIT_RADIUS,
                       PLAYER_W, PLAY_BOTTOM, PLAY_TOP, SIDE_ENEMY_SPEED,
                       SIDE_WARN_SECONDS, Bullet, Difficulty, Enemy,
                       HighScoreStore, PlaneGame, _circle_hit)

FAILED = []
STEP = 1.0 / 60.0

START_CODE = "P11111"        # X11111 with P thumb
OPEN_UP_CODE = "P01111"      # Xx1111 with the thumb folded -> up, no start
LEFT_CODE = "P10000"
RIGHT_CODE = "N10000"
FIST_CODE = "P00000"


def check(name, ok, detail=""):
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             " - " + detail if detail else ""))
    if not ok:
        FAILED.append(name)


def make_store(folder):
    return HighScoreStore(os.path.join(folder, "highscore.json"),
                          legacy_path=os.path.join(folder, "data.txt"))


def new_game(folder, seed=1):
    return PlaneGame(seed=seed, highscore=make_store(folder))


def advance(game, seconds, code=None, direction=None, step=STEP):
    for _ in range(int(round(seconds / step))):
        game.update(step, code, direction)


def place_enemy_on_player(game, kind="top"):
    """Put an enemy exactly on the player's hit circle."""
    player_x, player_y = game.player.center
    return Enemy(kind, player_x - ENEMY_W / 2.0, player_y - ENEMY_H / 2.0,
                 vy=0.0)


def game_over(game):
    """Force a game over by colliding a single enemy with the player."""
    if game.state != PlaneGame.PLAYING:
        game.start()
    game.chances = 1
    game.invincible_timer = 0.0
    game.enemies = [place_enemy_on_player(game)]
    game.update(STEP, None)
    return game.state == PlaneGame.OVER


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------
def test_difficulty():
    d = Difficulty()

    d.at(0.0)
    check("0 s: A = 1.5 s", abs(d.spawn_interval() - 1.5) < 1e-9,
          "{:.3f}".format(d.spawn_interval()))
    check("0 s: B = 120 px/s", abs(d.fall_speed() - 120.0) < 1e-9,
          "{:.3f}".format(d.fall_speed()))
    check("0 s: C = 8 s", abs(d.side_interval() - 8.0) < 1e-9,
          "{:.3f}".format(d.side_interval()))
    check("0 s: stage 1", d.stage == 0, str(d.stage))
    check("0 s: C jitter = 1.5 s", abs(d.side_interval_jitter() - 1.5) < 1e-9)

    d.at(10.0)
    check("10 s: A = 1.25 s", abs(d.spawn_interval() - 1.25) < 1e-9,
          "{:.3f}".format(d.spawn_interval()))
    check("10 s: B = 135 px/s", abs(d.fall_speed() - 135.0) < 1e-9,
          "{:.3f}".format(d.fall_speed()))
    check("10 s: C = 8 s", abs(d.side_interval() - 8.0) < 1e-9)

    d.at(20.0)
    check("20 s: A = 1.0 s", abs(d.spawn_interval() - 1.0) < 1e-9,
          "{:.3f}".format(d.spawn_interval()))
    check("20 s: B = 150 px/s", abs(d.fall_speed() - 150.0) < 1e-9,
          "{:.3f}".format(d.fall_speed()))
    check("20 s: C = 8 s", abs(d.side_interval() - 8.0) < 1e-9)
    check("20 s: stage 2", d.stage == 1, str(d.stage))
    check("20 s: C jitter switches to 1.0 s",
          abs(d.side_interval_jitter() - 1.0) < 1e-9)

    d.at(30.0)
    check("30 s: A = 0.9 s", abs(d.spawn_interval() - 0.9) < 1e-9,
          "{:.3f}".format(d.spawn_interval()))
    check("30 s: B = 170 px/s", abs(d.fall_speed() - 170.0) < 1e-9,
          "{:.3f}".format(d.fall_speed()))
    check("30 s: C = 7 s", abs(d.side_interval() - 7.0) < 1e-9,
          "{:.3f}".format(d.side_interval()))

    d.at(40.0)
    check("40 s: A = 0.8 s", abs(d.spawn_interval() - 0.8) < 1e-9,
          "{:.3f}".format(d.spawn_interval()))
    check("40 s: B = 190 px/s", abs(d.fall_speed() - 190.0) < 1e-9,
          "{:.3f}".format(d.fall_speed()))
    check("40 s: C = 6 s", abs(d.side_interval() - 6.0) < 1e-9,
          "{:.3f}".format(d.side_interval()))
    check("40 s: stage 3", d.stage == 2, str(d.stage))
    check("40 s: C jitter = 1.0 s",
          abs(d.side_interval_jitter() - 1.0) < 1e-9)

    d.at(90.0)
    check("90 s: difficulty stays constant",
          (abs(d.spawn_interval() - 0.8) < 1e-9 and
           abs(d.fall_speed() - 190.0) < 1e-9 and
           abs(d.side_interval() - 6.0) < 1e-9))
    check("90 s: progress is clamped to 1", d.progress == 1.0)

    # Monotonic ramp, never easier.
    previous = None
    monotonic = True
    for elapsed in range(0, 61):
        d.at(float(elapsed))
        values = (d.spawn_interval(), d.fall_speed(), d.side_interval())
        if previous is not None:
            if (values[0] > previous[0] + 1e-9 or
                    values[1] < previous[1] - 1e-9 or
                    values[2] > previous[2] + 1e-9):
                monotonic = False
        previous = values
    check("difficulty is monotonic over 0-60 s", monotonic)


def test_jitter_ranges():
    with tempfile.TemporaryDirectory() as folder:
        game = new_game(folder, seed=5)

        game.difficulty.at(0.0)
        intervals = [game._next_interval() for _ in range(300)]
        speeds = [game._roll_speed() for _ in range(300)]
        sides = [game._next_side_interval() for _ in range(300)]
        check("0 s: A in 1.5 +- 0.2 s",
              min(intervals) >= 1.3 - 1e-9 and max(intervals) <= 1.7 + 1e-9,
              "{:.3f}..{:.3f}".format(min(intervals), max(intervals)))
        check("0 s: B in 120 +- 20 px/s",
              min(speeds) >= 100.0 - 1e-9 and max(speeds) <= 140.0 + 1e-9,
              "{:.1f}..{:.1f}".format(min(speeds), max(speeds)))
        check("0 s: C in 8 +- 1.5 s",
              min(sides) >= 6.5 - 1e-9 and max(sides) <= 9.5 + 1e-9,
              "{:.3f}..{:.3f}".format(min(sides), max(sides)))

        game.difficulty.at(40.0)
        intervals = [game._next_interval() for _ in range(300)]
        speeds = [game._roll_speed() for _ in range(300)]
        sides = [game._next_side_interval() for _ in range(300)]
        check("40 s: A in 0.8 +- 0.2 s",
              min(intervals) >= 0.6 - 1e-9 and max(intervals) <= 1.0 + 1e-9,
              "{:.3f}..{:.3f}".format(min(intervals), max(intervals)))
        check("40 s: B in 190 +- 20 px/s",
              min(speeds) >= 170.0 - 1e-9 and max(speeds) <= 210.0 + 1e-9,
              "{:.1f}..{:.1f}".format(min(speeds), max(speeds)))
        check("40 s: C in 6 +- 1.0 s",
              min(sides) >= 5.0 - 1e-9 and max(sides) <= 7.0 + 1e-9,
              "{:.3f}..{:.3f}".format(min(sides), max(sides)))

        # Every spawned top enemy gets its own speed.
        game.start()
        game.elapsed = 40.0
        game.difficulty.at(40.0)
        game.enemies = []
        for _ in range(20):
            game._spawn_top_enemy()
        speeds = [enemy.vy for enemy in game.enemies]
        check("each top enemy has an individually rolled speed",
              len(set(speeds)) > 1 and
              all(170.0 - 1e-9 <= value <= 210.0 + 1e-9 for value in speeds))


# ---------------------------------------------------------------------------
# Hold to start / restart
# ---------------------------------------------------------------------------
def test_hold_to_start():
    with tempfile.TemporaryDirectory() as folder:
        game = new_game(folder)
        check("starts on the READY screen", game.state == PlaneGame.READY)

        advance(game, 0.5, START_CODE)
        check("0.5 s of X11111 is not enough", game.state == PlaneGame.READY,
              game.state)
        check("hold bar at 0.5 s", abs(game.hold_progress - 0.5 / HOLD_SECONDS)
              < 0.05, "{:.2f}".format(game.hold_progress))

        advance(game, 0.25, START_CODE)
        check("0.7 s of X11111 starts the round",
              game.state == PlaneGame.PLAYING, game.state)
        check("a new round has 3 chances", game.chances == MAX_CHANCES)
        check("a new round starts at 0 s", game.elapsed < 0.1)

        # Releasing the pose drains the bar again.
        game.restart()
        advance(game, 0.05, None)          # arm the pose detection
        advance(game, 0.5, START_CODE)
        advance(game, 0.1, None)
        check("releasing the pose drains the hold bar",
              game.hold_progress < 0.6,
              "{:.2f}".format(game.hold_progress))
        check("the round did not start", game.state == PlaneGame.READY)

        # The thumb-folded open hand moves up but never starts a round.
        game2 = new_game(folder, seed=2)
        advance(game2, 2.0, OPEN_UP_CODE)
        check("Xx1111 with a folded thumb does not start",
              game2.state == PlaneGame.READY, game2.state)

        # Other poses do not start either.
        for code in (FIST_CODE, LEFT_CODE, RIGHT_CODE, "P01000", None):
            game3 = new_game(folder, seed=3)
            advance(game3, 1.5, code)
            check("{!r} does not start a round".format(code),
                  game3.state == PlaneGame.READY)


def test_restart():
    with tempfile.TemporaryDirectory() as folder:
        game = new_game(folder)
        check("game over", game_over(game))
        check("game over keeps the score", game.last_score == game.score)

        # The pose that was held through the game over must not restart
        # immediately: the hand has to be lowered once.
        advance(game, 1.5, START_CODE)
        check("holding X11111 right after game over does not restart",
              game.state == PlaneGame.OVER, game.state)

        advance(game, 0.05, None)
        advance(game, 0.8, START_CODE)
        check("release + 0.7 s of X11111 restarts directly",
              game.state == PlaneGame.PLAYING, game.state)

        # The restart button goes back to the READY screen.
        check("second game over", game_over(game))
        game.restart()
        check("restart() returns to the READY screen",
              game.state == PlaneGame.READY)
        advance(game, 1.0, START_CODE)
        check("the button alone does not start a round",
              game.state == PlaneGame.READY)
        advance(game, 0.05, None)
        advance(game, 0.8, START_CODE)
        check("the pose still starts the next round",
              game.state == PlaneGame.PLAYING)


# ---------------------------------------------------------------------------
# Chances / invincibility / collision
# ---------------------------------------------------------------------------
def test_hit_boxes():
    check("player radius covers half a wing",
          abs(PLAYER_HIT_RADIUS - PLAYER_W / 4.0) < 1e-9,
          str(PLAYER_HIT_RADIUS))
    check("player circle is smaller than a wing span",
          PLAYER_HIT_RADIUS < PLAYER_W / 2.0)
    check("enemy radius derived the same way",
          abs(ENEMY_HIT_RADIUS - ENEMY_W / 4.0) < 1e-9,
          str(ENEMY_HIT_RADIUS))
    radius = PLAYER_HIT_RADIUS + ENEMY_HIT_RADIUS
    check("circles just outside the sum do not collide",
          not _circle_hit(0.0, 0.0, PLAYER_HIT_RADIUS,
                          radius + 1.0, 0.0, ENEMY_HIT_RADIUS))
    check("overlapping circles collide",
          _circle_hit(0.0, 0.0, PLAYER_HIT_RADIUS,
                      radius - 1.0, 0.0, ENEMY_HIT_RADIUS))


def test_chances_and_invincibility():
    with tempfile.TemporaryDirectory() as folder:
        game = new_game(folder)
        game.start()
        game.enemies = [place_enemy_on_player(game)]
        game.update(STEP, None)
        check("a collision costs one chance", game.chances == 2,
              str(game.chances))
        check("the enemy is removed", not game.enemies)
        check("a collision starts 3 s of invincibility",
              abs(game.invincible_timer - INVINCIBLE_SECONDS) < 0.02,
              "{:.2f}".format(game.invincible_timer))
        check("invincibility progress is near 1",
              game.invincible_progress > 0.98)

        game.enemies = [place_enemy_on_player(game)]
        game.update(STEP, None)
        check("no chance is lost while invincible", game.chances == 2,
              str(game.chances))
        check("the enemy touching an invincible plane stays",
              len(game.enemies) == 1, str(len(game.enemies)))

        # The invincibility runs out after 3 s (clear the enemy first, or it
        # would immediately cost the next chance and restart the timer).
        game.enemies = []
        advance(game, 3.1, None)
        check("invincibility expires", game.invincible_timer == 0.0)
        game.enemies = [place_enemy_on_player(game)]
        game.update(STEP, None)
        check("a hit after the invincibility costs a chance",
              game.chances == 1, str(game.chances))

        game.enemies = [place_enemy_on_player(game)]
        game.invincible_timer = 0.0
        game.update(STEP, None)
        check("the third hit ends the round",
              game.state == PlaneGame.OVER and game.chances == 0,
              "{} {}".format(game.state, game.chances))

        check("the hit flash is running", game.hit_flash > 0.0)


def test_bullets_and_score():
    with tempfile.TemporaryDirectory() as folder:
        game = new_game(folder)
        game.start()
        advance(game, 0.3, None)
        check("the plane fires automatically", len(game.bullets) >= 1,
              str(len(game.bullets)))

        enemy = Enemy("top", 200.0, 300.0, vy=0.0)
        game.enemies = [enemy]
        game.bullets = [Bullet(200.0 + ENEMY_W / 2.0 - 4.5,
                               300.0 + ENEMY_H / 2.0 - 10.5)]
        game._resolve_collisions()
        check("a bullet destroys the enemy it hits",
              not game.enemies and not game.bullets)
        check("destroying an enemy scores one point", game.score == 1,
              str(game.score))

        # The explosion was staged and the bullet is gone.
        check("a destroyed enemy leaves an explosion", len(game.explosions) == 1)

        # Off-screen enemies are removed without costing a chance.
        game.enemies = [Enemy("top", 100.0, PLAY_BOTTOM + 10.0, vy=0.0),
                        Enemy("side", -ENEMY_W - 20.0, 300.0,
                              vx=SIDE_ENEMY_SPEED)]
        chances = game.chances
        game.update(STEP, None)
        check("enemies that left the screen are dropped", not game.enemies)
        check("enemies leaving the screen cost no chance",
              game.chances == chances)


def test_player_movement_and_bounds():
    with tempfile.TemporaryDirectory() as folder:
        game = new_game(folder)
        game.start()

        game.player.x = 0.0
        game.update(0.1, LEFT_CODE)
        check("the plane stops at the left edge", game.player.x == 0.0)

        game.player.x = game.width - game.player.w
        game.update(0.1, RIGHT_CODE)
        check("the plane stops at the right edge",
              game.player.x == game.width - game.player.w)

        game.player.y = PLAY_TOP
        game.update(0.1, OPEN_UP_CODE)
        check("the plane stops below the HUD", game.player.y == PLAY_TOP)

        game.player.y = PLAY_BOTTOM - game.player.h
        game.update(0.1, FIST_CODE)
        check("the plane stops above the bottom panel",
              game.player.y == PLAY_BOTTOM - game.player.h)

        start_y = game.player.y
        advance(game, 0.2, OPEN_UP_CODE, step=STEP)
        check("Xx1111 moves the plane up", game.player.y < start_y,
              "{:.1f} -> {:.1f}".format(start_y, game.player.y))


# ---------------------------------------------------------------------------
# Side enemies
# ---------------------------------------------------------------------------
def test_side_enemy():
    with tempfile.TemporaryDirectory() as folder:
        game = new_game(folder)
        game.start()

        # Player near the left edge -> the side enemy enters from the right.
        game.player.x = 10.0
        game.side_timer = 0.01
        game.enemies = []
        game.bullets = []
        game.update(0.02, None)
        check("a side warning was announced", len(game.warnings) == 1,
              str(len(game.warnings)))
        warning = game.warnings[0]
        check("a player near the left is attacked from the right",
              warning.side == "R", warning.side)
        check("the warning lasts 3 s",
              abs(warning.ttl - (SIDE_WARN_SECONDS - 0.02)) < 0.03,
              "{:.2f}".format(warning.ttl))
        check("the countdown starts at 3", warning.remaining == 3,
              str(warning.remaining))
        entry_y = warning.y
        check("the entry row is inside the play area",
              PLAY_TOP <= entry_y <= PLAY_BOTTOM - ENEMY_H,
              "{:.1f}".format(entry_y))
        check("the entry row aims at the player's row",
              abs(entry_y - (game.player.center[1] - ENEMY_H / 2.0)) < 1e-6,
              "{:.1f} vs {:.1f}".format(
                  entry_y, game.player.center[1] - ENEMY_H / 2.0))

        advance(game, 1.0, None)
        check("countdown shows 2", game.warnings[0].remaining == 2,
              str(game.warnings[0].remaining))
        advance(game, 1.0, None)
        check("countdown shows 1", game.warnings[0].remaining == 1,
              str(game.warnings[0].remaining))

        # The enemy appears at the announced row once the countdown ends.
        advance(game, 1.1, None)
        check("the warning disappears", not game.warnings)
        side = [enemy for enemy in game.enemies if enemy.kind == "side"]
        check("exactly one side enemy appeared", len(side) == 1, str(len(side)))
        if side:
            enemy = side[0]
            check("the side enemy enters from the right edge",
                  enemy.x > game.width - 20.0, "{:.1f}".format(enemy.x))
            check("the side enemy flies left", enemy.vx < 0.0,
                  str(enemy.vx))
            check("the side enemy flies very slowly",
                  abs(abs(enemy.vx) - SIDE_ENEMY_SPEED) < 1e-9, str(enemy.vx))
            check("the side enemy enters at the announced row",
                  abs(enemy.y - entry_y) < 1e-6,
                  "{:.2f} vs {:.2f}".format(enemy.y, entry_y))

        # Player near the right edge -> the side enemy enters from the left.
        game.player.x = game.width - game.player.w
        game.player.y = 260.0
        game.side_timer = 0.01
        game.update(0.02, None)
        check("a player near the right is attacked from the left",
              game.warnings and game.warnings[-1].side == "L",
              game.warnings[-1].side if game.warnings else "no warning")
        check("the row follows the player's current position",
              abs(game.warnings[-1].y -
                  (game.player.center[1] - ENEMY_H / 2.0)) < 1e-6,
              "{:.1f} vs {:.1f}".format(
                  game.warnings[-1].y,
                  game.player.center[1] - ENEMY_H / 2.0))

        # The side enemy crosses the screen at a constant height and leaves.
        game.warnings = []
        game.enemies = [Enemy("side", game.width - 20.0, 300.0,
                              vx=-SIDE_ENEMY_SPEED)]
        advance(game, 9.5, None)
        check("the side enemy leaves the screen",
              not [enemy for enemy in game.enemies if enemy.kind == "side"])


# ---------------------------------------------------------------------------
# High score
# ---------------------------------------------------------------------------
def test_high_score():
    with tempfile.TemporaryDirectory() as folder:
        store = make_store(folder)
        game = new_game(folder)
        check("a fresh record starts at 0", game.best_score == 0)

        game.start()
        game.score = 7
        check("game over", game_over(game))
        record = store.load()
        check("the finished game stores the best score",
              record["best"] == 7, str(record))
        check("the finished game is counted", record["games"] == 1,
              str(record))
        check("the last score is stored", record["last"] == 7)
        check("the game instance knows the new best", game.best_score == 7)
        check("a new record is flagged", game.new_record)

        # A lower score does not lower the best.
        game.start()
        game.score = 3
        check("game over again", game_over(game))
        record = store.load()
        check("a lower score keeps the best", record["best"] == 7,
              str(record))
        check("the second game is counted", record["games"] == 2)
        check("a lower score is not a record", not game.new_record)

        # A reloaded game reads the stored best.
        reloaded = new_game(folder)
        check("the best score is restored on startup",
              reloaded.best_score == 7, str(reloaded.best_score))


def test_legacy_high_score():
    with tempfile.TemporaryDirectory() as folder:
        legacy = os.path.join(folder, "data.txt")
        with open(legacy, "w", encoding="utf-8") as handle:
            handle.write("25\n")
        store = HighScoreStore(os.path.join(folder, "highscore.json"),
                               legacy_path=legacy)
        check("the old data.txt score is imported", store.load()["best"] == 25,
              str(store.load()))

        with open(legacy, "w", encoding="utf-8") as handle:
            handle.write("not a number\n")
        check("a broken data.txt does not crash the game",
              store.load()["best"] == 0, str(store.load()))

        # Once highscore.json exists it wins over the legacy file.
        store.save({"best": 40, "last": 12, "games": 3})
        with open(legacy, "w", encoding="utf-8") as handle:
            handle.write("25\n")
        check("highscore.json wins over data.txt",
              store.load()["best"] == 40, str(store.load()))


def main():
    test_difficulty()
    test_jitter_ranges()
    test_hold_to_start()
    test_restart()
    test_hit_boxes()
    test_chances_and_invincibility()
    test_bullets_and_score()
    test_player_movement_and_bounds()
    test_side_enemy()
    test_high_score()
    test_legacy_high_score()
    print()
    if FAILED:
        print("RESULT: {} failed: {}".format(len(FAILED), ", ".join(FAILED)))
        return 1
    print("RESULT: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
