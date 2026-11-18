# -*- coding: utf-8 -*-
"""Pure game logic of the hand-gesture plane game (no pygame, no camera).

The scene and the hand recogniser live elsewhere; this module only knows about
the plane, the enemies, the difficulty ramp and the game state machine, so it
can be unit tested without a display, a camera or MediaPipe.

Rules implemented here (see ``工作汇报.md`` for the full description):

* the player plane moves with the recognised direction and fires automatically;
* three chances; every collision with an enemy costs one chance and grants
  3 s of invincibility (the scene blinks the plane and draws a yellow
  countdown ring around it);
* the round ends when the third chance is gone;
* a round only starts after the ``X11111`` pose (the whole hand open, thumb
  included) was held for 0.7 s, and the same pose restarts from the game over
  screen;
* difficulty is defined by three quantities that change in two 20 s stages:

  ==========  =================================  ==========================
  quantity    stage 1 (0-20 s)                   stage 2 (20-40 s, then
                                                 constant)
  ==========  =================================  ==========================
  A           top spawn interval 1.5 -> 1.0 s    1.0 -> 0.8 s
              (+-0.2 s per spawn)                (+-0.2 s per spawn)
  B           fall speed 120 -> 150 px/s         150 -> 190 px/s
              (+-20 px/s per enemy)              (+-20 px/s per enemy)
  C           side spawn interval 8 s (+-1.5 s)  8 -> 6 s (+-1.0 s)
  ==========  =================================  ==========================

* the new side enemies fly in horizontally at a very low speed; their entry
  point is announced 3 s in advance.  A side enemy enters from the right when
  the player is near the left edge at spawn time, and from the left otherwise.
"""

from __future__ import absolute_import

import json
import math
import random

from gesture_code import direction_of, is_start_pose

# ---------------------------------------------------------------------------
# Canvas geometry.  The scene is laid out for 480x800 (the size of the bundled
# artwork) and displayed in a smaller window, like the reference rhythm game.
# ---------------------------------------------------------------------------
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 800
#: The window is the 480x800 canvas scaled down to 87.5 % (420x700).
WINDOW_SCALE = 0.875
WINDOW_SIZE = (int(SCREEN_WIDTH * WINDOW_SCALE),
               int(SCREEN_HEIGHT * WINDOW_SCALE))

#: Top HUD strip and bottom gesture strip of the playing screen.
HUD_HEIGHT = 84
BOTTOM_PANEL_HEIGHT = 56
PLAY_TOP = HUD_HEIGHT
PLAY_BOTTOM = SCREEN_HEIGHT - BOTTOM_PANEL_HEIGHT

# ---------------------------------------------------------------------------
# Player / bullets.
# ---------------------------------------------------------------------------
PLAYER_W, PLAYER_H = 102, 126
PLAYER_SPEED = 400.0                    # px/s
#: Circular hit box of the player.  The sprite is 102 px wide, i.e. the wings
#: span 51 px to each side; the circle only covers *half* of one wing, so its
#: radius is 102 / 4 = 25.5 px.
PLAYER_HIT_RADIUS = PLAYER_W / 4.0

BULLET_W, BULLET_H = 9, 21
BULLET_SPEED = 620.0                    # px/s
BULLET_RADIUS = 4.5
FIRE_INTERVAL = 0.18                    # s between two automatic shots

# ---------------------------------------------------------------------------
# Enemies.
# ---------------------------------------------------------------------------
ENEMY_W, ENEMY_H = 57, 43
#: Circular hit box of an enemy, derived the same way as the player's:
#: 57 / 4 = 14.25 px (half of one wing half-span).
ENEMY_HIT_RADIUS = ENEMY_W / 4.0
#: Side enemies fly in horizontally at a deliberately low speed.
SIDE_ENEMY_SPEED = 55.0                 # px/s
#: The entry point of a side enemy is announced this long before it appears.
SIDE_WARN_SECONDS = 3.0
#: Vertical margin kept between a side entry point and the play area border.
SIDE_ENTRY_MARGIN = 12.0

# ---------------------------------------------------------------------------
# Round rules.
# ---------------------------------------------------------------------------
MAX_CHANCES = 3
HOLD_SECONDS = 0.7                      # X11111 has to be held this long
INVINCIBLE_SECONDS = 3.0
HIT_FLASH_SECONDS = 0.5
EXPLOSION_SECONDS = 0.4
POPUP_SECONDS = 0.8
#: A round always starts with a short grace period before the first enemy.
START_GRACE_SECONDS = 0.6

# ---------------------------------------------------------------------------
# Difficulty ramp: two 20 s stages, then constant.
# ---------------------------------------------------------------------------
RAMP_SECONDS = 40.0
STAGE_1_END = 20.0
STAGE_2_END = 40.0

#: A: top spawn interval (seconds) at the start / 20 s / 40 s.
SPAWN_INTERVAL_START, SPAWN_INTERVAL_MID, SPAWN_INTERVAL_END = 1.5, 1.0, 0.8
SPAWN_INTERVAL_JITTER = 0.2
#: B: fall speed (px/s) at the start / 20 s / 40 s.
FALL_SPEED_START, FALL_SPEED_MID, FALL_SPEED_END = 120.0, 150.0, 190.0
FALL_SPEED_JITTER = 20.0
#: C: side spawn interval (seconds) at the start / 20 s / 40 s.
SIDE_INTERVAL_START, SIDE_INTERVAL_MID, SIDE_INTERVAL_END = 8.0, 8.0, 6.0
SIDE_INTERVAL_JITTER_EARLY = 1.5
SIDE_INTERVAL_JITTER_LATE = 1.0
#: Safety floor for the rolled intervals.
MIN_SPAWN_INTERVAL = 0.35
MIN_SIDE_INTERVAL = 3.0


def _lerp(start, end, t):
    """Linear interpolation between *start* and *end* (t in 0..1)."""
    return start + (end - start) * max(0.0, min(1.0, t))


def _circle_hit(x1, y1, r1, x2, y2, r2):
    """True when two circles overlap."""
    dx = x1 - x2
    dy = y1 - y2
    radius = r1 + r2
    return dx * dx + dy * dy <= radius * radius


# ---------------------------------------------------------------------------
# High score persistence
# ---------------------------------------------------------------------------
class HighScoreStore(object):
    """Tiny JSON persistence for the best score (never crashes the game).

    ``data.txt`` of the previous version only contained the best score; when
    ``highscore.json`` does not exist yet that value is used as the starting
    best score so the old record is not lost.
    """

    def __init__(self, path="highscore.json", legacy_path="data.txt"):
        self.path = path
        self.legacy_path = legacy_path
        self.default = {"best": 0, "last": 0, "games": 0}

    def load(self):
        data = dict(self.default)
        loaded = None
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception:
            loaded = None
        if isinstance(loaded, dict):
            for key in data:
                value = loaded.get(key)
                if isinstance(value, (int, float)) and value >= 0:
                    data[key] = int(value)
            return data
        legacy = self._load_legacy()
        if legacy:
            data["best"] = legacy
        return data

    def _load_legacy(self):
        """Read the old plain-text high score file (``data.txt``)."""
        try:
            with open(self.legacy_path, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
            value = int(float(text))
        except Exception:
            return 0
        return max(0, value)

    def save(self, data):
        try:
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Difficulty
# ---------------------------------------------------------------------------
class Difficulty(object):
    """Two-stage difficulty ramp (see the module docstring).

    ``at`` is called with the elapsed round time; the three quantities are
    read per spawn, and the random fluctuation is rolled by the game so every
    enemy gets its own speed / every spawn its own interval.
    """

    def __init__(self, ramp_seconds=RAMP_SECONDS):
        self.ramp_seconds = float(ramp_seconds)
        self.elapsed = 0.0

    def at(self, elapsed):
        self.elapsed = max(0.0, float(elapsed))
        return self.elapsed

    @property
    def progress(self):
        """0..1 over the 40 s ramp (stays at 1 afterwards)."""
        if self.ramp_seconds <= 0:
            return 1.0
        return max(0.0, min(1.0, self.elapsed / self.ramp_seconds))

    @property
    def stage(self):
        """0 = 0-20 s, 1 = 20-40 s, 2 = after 40 s (constant)."""
        if self.elapsed < STAGE_1_END:
            return 0
        if self.elapsed < STAGE_2_END:
            return 1
        return 2

    @property
    def stage_t(self):
        """Progress (0..1) inside the current stage."""
        if self.elapsed < STAGE_1_END:
            return self.elapsed / STAGE_1_END
        if self.elapsed < STAGE_2_END:
            return ((self.elapsed - STAGE_1_END) /
                    (STAGE_2_END - STAGE_1_END))
        return 1.0

    # -- A: top spawn interval -----------------------------------------
    def spawn_interval(self):
        """Base interval (s) between two top enemies."""
        t = self.stage_t
        if self.stage == 0:
            return _lerp(SPAWN_INTERVAL_START, SPAWN_INTERVAL_MID, t)
        if self.stage == 1:
            return _lerp(SPAWN_INTERVAL_MID, SPAWN_INTERVAL_END, t)
        return SPAWN_INTERVAL_END

    # -- B: fall speed --------------------------------------------------
    def fall_speed(self):
        """Base fall speed (px/s) of a top enemy."""
        t = self.stage_t
        if self.stage == 0:
            return _lerp(FALL_SPEED_START, FALL_SPEED_MID, t)
        if self.stage == 1:
            return _lerp(FALL_SPEED_MID, FALL_SPEED_END, t)
        return FALL_SPEED_END

    # -- C: side spawn interval ----------------------------------------
    def side_interval(self):
        """Base interval (s) between two side-enemy warnings."""
        t = self.stage_t
        if self.stage == 0:
            return _lerp(SIDE_INTERVAL_START, SIDE_INTERVAL_MID, t)
        if self.stage == 1:
            return _lerp(SIDE_INTERVAL_MID, SIDE_INTERVAL_END, t)
        return SIDE_INTERVAL_END

    def side_interval_jitter(self):
        """Half-width (s) of the side interval fluctuation."""
        if self.stage == 0:
            return SIDE_INTERVAL_JITTER_EARLY
        return SIDE_INTERVAL_JITTER_LATE

    @property
    def label(self):
        """Short stage label for the HUD."""
        return ("热身", "加速", "极限")[self.stage]


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------
class Player(object):
    """The player plane (position is the top-left corner, in pixels)."""

    __slots__ = ("x", "y", "w", "h", "radius")

    def __init__(self, x, y, w=PLAYER_W, h=PLAYER_H, radius=PLAYER_HIT_RADIUS):
        self.x = float(x)
        self.y = float(y)
        self.w = int(w)
        self.h = int(h)
        self.radius = float(radius)

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def rect(self):
        return (int(round(self.x)), int(round(self.y)), self.w, self.h)


class Bullet(object):
    """A bullet fired by the player (moves straight up)."""

    __slots__ = ("x", "y", "w", "h", "speed", "radius", "alive")

    def __init__(self, x, y, w=BULLET_W, h=BULLET_H, speed=BULLET_SPEED,
                 radius=BULLET_RADIUS):
        self.x = float(x)
        self.y = float(y)
        self.w = int(w)
        self.h = int(h)
        self.speed = float(speed)
        self.radius = float(radius)
        self.alive = True

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def rect(self):
        return (int(round(self.x)), int(round(self.y)), self.w, self.h)


class Enemy(object):
    """An enemy plane.

    ``kind`` is ``"top"`` for the classic enemies falling from the top edge
    (speed rolled individually) and ``"side"`` for the horizontal ones.
    """

    __slots__ = ("kind", "x", "y", "w", "h", "vx", "vy", "radius", "alive")

    def __init__(self, kind, x, y, vx=0.0, vy=0.0, w=ENEMY_W, h=ENEMY_H,
                 radius=ENEMY_HIT_RADIUS):
        self.kind = kind
        self.x = float(x)
        self.y = float(y)
        self.w = int(w)
        self.h = int(h)
        self.vx = float(vx)
        self.vy = float(vy)
        self.radius = float(radius)
        self.alive = True

    @property
    def center(self):
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def rect(self):
        return (int(round(self.x)), int(round(self.y)), self.w, self.h)

    @property
    def moving_right(self):
        return self.kind == "side" and self.vx > 0.0


class SideWarning(object):
    """Countdown marker announcing where a side enemy will fly in."""

    __slots__ = ("side", "y", "ttl")

    def __init__(self, side, y, ttl=SIDE_WARN_SECONDS):
        self.side = side          # "L" enters from the left, "R" from the right
        self.y = float(y)         # top edge of the enemy when it appears
        self.ttl = float(ttl)

    @property
    def remaining(self):
        """Whole seconds left (3, 2, 1) for the on-screen countdown."""
        return max(1, int(math.ceil(self.ttl - 1e-9)))


class Explosion(object):
    """Short-lived explosion sprite at a fixed position."""

    __slots__ = ("x", "y", "age", "ttl")

    def __init__(self, x, y, ttl=EXPLOSION_SECONDS):
        self.x = float(x)
        self.y = float(y)
        self.age = 0.0
        self.ttl = float(ttl)

    @property
    def progress(self):
        if self.ttl <= 0:
            return 1.0
        return max(0.0, min(1.0, self.age / self.ttl))


class Popup(object):
    """Floating "+1" / "机会 -1" text."""

    __slots__ = ("x", "y", "text", "kind", "age", "ttl")

    def __init__(self, x, y, text, kind="score", ttl=POPUP_SECONDS):
        self.x = float(x)
        self.y = float(y)
        self.text = text
        self.kind = kind          # "score" or "danger"
        self.age = 0.0
        self.ttl = float(ttl)


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------
class PlaneGame(object):
    """State machine, spawner, collisions and score of the plane game."""

    READY = "ready"
    PLAYING = "playing"
    OVER = "over"

    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT,
                 play_top=PLAY_TOP, play_bottom=PLAY_BOTTOM,
                 max_chances=MAX_CHANCES, hold_seconds=HOLD_SECONDS,
                 invincible_seconds=INVINCIBLE_SECONDS,
                 seed=None, highscore=None, difficulty=None):
        self.width = width
        self.height = height
        self.play_top = play_top
        self.play_bottom = play_bottom
        self.max_chances = int(max_chances)
        self.hold_seconds = float(hold_seconds)
        self.invincible_seconds = float(invincible_seconds)
        self.rng = random.Random(seed)
        self.difficulty = difficulty or Difficulty()

        self.highscores = highscore or HighScoreStore()
        self.record = self.highscores.load()
        self.best_score = self.record["best"]

        self.time = 0.0
        self.events = []
        self.last_code = None
        self.last_direction = "pause"
        self.reset()

    # -- state ----------------------------------------------------------
    def reset(self):
        """Back to the start screen (holding X11111 starts a round)."""
        self.state = self.READY
        self.score = 0
        self.chances = self.max_chances
        self.elapsed = 0.0
        self.survived = 0.0
        self.hold_time = 0.0
        #: The pose only counts after a frame *without* the pose was seen, so
        #: a hand that is already held up cannot restart the game by accident.
        self.armed = True
        self.invincible_timer = 0.0
        self.hit_flash = 0.0
        self.difficulty.at(0.0)
        self.player = Player(self.width // 2 - PLAYER_W // 2,
                             self.play_bottom - PLAYER_H - 10)
        self.bullets = []
        self.enemies = []
        self.warnings = []
        self.explosions = []
        self.popups = []
        self.spawn_timer = 0.0
        self.side_timer = 0.0
        self.fire_timer = 0.0
        self.new_record = False
        self.last_score = 0
        self.last_code = None
        self.last_direction = "pause"
        self.events = []

    def start(self):
        """Start a fresh round (the caller decided the pose was held)."""
        self.reset()
        self.state = self.PLAYING
        self.spawn_timer = START_GRACE_SECONDS + self._next_interval()
        self.side_timer = self._next_side_interval()
        self.fire_timer = 0.05
        self.events = []

    def restart(self):
        """Restart button: back to the start screen.

        The next round still has to be started by holding ``X11111`` again,
        exactly like the first round.
        """
        self.reset()
        self.armed = False

    @property
    def hold_progress(self):
        if self.hold_seconds <= 0:
            return 1.0
        return max(0.0, min(1.0, self.hold_time / self.hold_seconds))

    @property
    def invincible_progress(self):
        """1.0 right after a hit, 0.0 when the invincibility ran out."""
        if self.invincible_seconds <= 0:
            return 0.0
        return max(0.0, min(1.0,
                            self.invincible_timer / self.invincible_seconds))

    @property
    def invincible(self):
        return self.invincible_timer > 0.0

    @property
    def stage(self):
        """1-based stage number for the HUD."""
        return self.difficulty.stage + 1

    @property
    def difficulty_value(self):
        return self.difficulty.progress

    # -- update ---------------------------------------------------------
    def update(self, dt, gesture=None, direction=None):
        """Advance the game by *dt* seconds.

        Args:
            gesture: object with a ``code`` attribute (a
                :class:`gesture_single.GestureState`), a 6 character code
                string, or ``None``.
            direction: optional direction override (arrow-key control); when
                ``None`` the direction is derived from the hand code.

        Returns:
            The list of events of this frame.
        """
        dt = max(0.0, min(float(dt), 0.1))
        self.time += dt
        self.events = []

        code = self._code_of(gesture)
        if direction is None:
            direction = direction_of(code)
        self.last_code = code
        self.last_direction = direction
        pose_held = is_start_pose(code)

        self._update_effects(dt)

        if self.state == self.READY:
            self._update_ready(dt, pose_held)
        elif self.state == self.PLAYING:
            self._update_playing(dt, direction)
        elif self.state == self.OVER:
            self._update_over(dt, pose_held)
        return self.events

    @staticmethod
    def _code_of(gesture):
        """Accept a GestureState, a plain code string or ``None``."""
        if gesture is None:
            return None
        if isinstance(gesture, str):
            return gesture or None
        return getattr(gesture, "code", None)

    def _update_effects(self, dt):
        self.hit_flash = max(0.0, self.hit_flash - dt)
        for explosion in self.explosions:
            explosion.age += dt
        self.explosions = [item for item in self.explosions
                           if item.age < item.ttl]
        for popup in self.popups:
            popup.age += dt
        self.popups = [item for item in self.popups if item.age < item.ttl]

    def _update_ready(self, dt, pose_held):
        if not pose_held:
            self.hold_time = max(0.0, self.hold_time - dt * 2.5)
            self.armed = True
            return
        if not self.armed:
            self.hold_time = 0.0
            return
        self.hold_time += dt
        if self.hold_time >= self.hold_seconds:
            self.start()
            self.events.append({"type": "start"})

    def _update_over(self, dt, pose_held):
        if not pose_held:
            self.hold_time = max(0.0, self.hold_time - dt * 2.5)
            self.armed = True
            return
        if not self.armed:
            self.hold_time = 0.0
            return
        self.hold_time += dt
        if self.hold_time >= self.hold_seconds:
            self.start()
            self.events.append({"type": "restart"})

    # -- playing --------------------------------------------------------
    def _update_playing(self, dt, direction):
        self.elapsed += dt
        self.survived = self.elapsed
        self.difficulty.at(self.elapsed)

        self._move_player(dt, direction)
        self._update_fire(dt)
        self._update_spawns(dt)
        self._update_warnings(dt)
        self._move_bullets(dt)
        self._move_enemies(dt)
        self._resolve_collisions()
        if self.state != self.PLAYING:
            return
        if self.invincible_timer > 0.0:
            self.invincible_timer = max(0.0, self.invincible_timer - dt)

    def _move_player(self, dt, direction):
        dx = dy = 0.0
        if direction == "left":
            dx = -PLAYER_SPEED * dt
        elif direction == "right":
            dx = PLAYER_SPEED * dt
        elif direction == "up":
            dy = -PLAYER_SPEED * dt
        elif direction == "down":
            dy = PLAYER_SPEED * dt
        player = self.player
        player.x = min(max(0.0, player.x + dx), self.width - player.w)
        player.y = min(max(float(self.play_top), player.y + dy),
                       self.play_bottom - player.h)

    def _update_fire(self, dt):
        self.fire_timer -= dt
        guard = 0
        while self.fire_timer <= 0.0 and guard < 4:
            self._fire()
            self.fire_timer += FIRE_INTERVAL
            guard += 1

    def _fire(self):
        player = self.player
        self.bullets.append(Bullet(player.center[0] - BULLET_W / 2.0,
                                   player.y + 10 - BULLET_H))

    def _update_spawns(self, dt):
        self.spawn_timer -= dt
        guard = 0
        while self.spawn_timer <= 0.0 and guard < 3:
            self._spawn_top_enemy()
            self.spawn_timer += self._next_interval()
            guard += 1

        self.side_timer -= dt
        if self.side_timer <= 0.0:
            self._spawn_side_warning()
            self.side_timer += self._next_side_interval()

    def _update_warnings(self, dt):
        still_waiting = []
        for warning in self.warnings:
            warning.ttl -= dt
            if warning.ttl <= 0.0:
                self._launch_side_enemy(warning)
            else:
                still_waiting.append(warning)
        self.warnings = still_waiting

    def _move_bullets(self, dt):
        alive = []
        for bullet in self.bullets:
            bullet.y -= bullet.speed * dt
            # A bullet is removed once it has left the play area at the top.
            if bullet.y + bullet.h > self.play_top:
                alive.append(bullet)
        self.bullets = alive

    def _move_enemies(self, dt):
        alive = []
        for enemy in self.enemies:
            if not enemy.alive:
                continue
            if enemy.kind == "top":
                enemy.y += enemy.vy * dt
                if enemy.y > self.play_bottom:
                    continue
            else:
                enemy.x += enemy.vx * dt
                if enemy.x + enemy.w < -4 or enemy.x > self.width + 4:
                    continue
            alive.append(enemy)
        self.enemies = alive

    # -- spawning -------------------------------------------------------
    def _next_interval(self):
        base = self.difficulty.spawn_interval()
        value = base + self.rng.uniform(-SPAWN_INTERVAL_JITTER,
                                        SPAWN_INTERVAL_JITTER)
        return max(MIN_SPAWN_INTERVAL, value)

    def _next_side_interval(self):
        base = self.difficulty.side_interval()
        jitter = self.difficulty.side_interval_jitter()
        value = base + self.rng.uniform(-jitter, jitter)
        return max(MIN_SIDE_INTERVAL, value)

    def _roll_speed(self):
        """Base fall speed of the stage plus a random fluctuation."""
        base = self.difficulty.fall_speed()
        return max(40.0, base + self.rng.uniform(-FALL_SPEED_JITTER,
                                                 FALL_SPEED_JITTER))

    def _spawn_top_enemy(self):
        x = self.rng.uniform(4.0, max(4.0, self.width - ENEMY_W - 4.0))
        enemy = Enemy("top", x, self.play_top - ENEMY_H, vy=self._roll_speed())
        self.enemies.append(enemy)
        return enemy

    def _spawn_side_warning(self):
        """Announce a side enemy, 3 s before it enters the screen.

        The side is chosen from the player position *at spawn time*: a player
        near the left edge is attacked from the right, otherwise from the left.
        """
        player_near_left = self.player.center[0] < self.width / 2.0
        side = "R" if player_near_left else "L"
        y = self.rng.uniform(
            self.play_top + SIDE_ENTRY_MARGIN,
            self.play_bottom - ENEMY_H - SIDE_ENTRY_MARGIN)
        warning = SideWarning(side, y)
        self.warnings.append(warning)
        self.events.append({"type": "warning", "side": side, "y": y})
        return warning

    def _launch_side_enemy(self, warning):
        if warning.side == "R":
            x = self.width + 2.0
            vx = -SIDE_ENEMY_SPEED
        else:
            x = -ENEMY_W - 2.0
            vx = SIDE_ENEMY_SPEED
        enemy = Enemy("side", x, warning.y, vx=vx, vy=0.0)
        self.enemies.append(enemy)
        self.events.append({"type": "side", "side": warning.side,
                            "y": warning.y})
        return enemy

    # -- collisions -----------------------------------------------------
    def _resolve_collisions(self):
        # Bullets vs. enemies.
        for bullet in self.bullets:
            if not bullet.alive:
                continue
            bx, by = bullet.center
            for enemy in self.enemies:
                if not enemy.alive:
                    continue
                ex, ey = enemy.center
                if _circle_hit(bx, by, bullet.radius, ex, ey, enemy.radius):
                    bullet.alive = False
                    enemy.alive = False
                    self._kill_enemy(enemy)
                    break
        self.bullets = [item for item in self.bullets if item.alive]
        self.enemies = [item for item in self.enemies if item.alive]

        # Player vs. enemies (ignored while invincible).
        if self.invincible_timer > 0.0:
            return
        px, py = self.player.center
        for enemy in self.enemies:
            ex, ey = enemy.center
            if _circle_hit(px, py, self.player.radius, ex, ey, enemy.radius):
                enemy.alive = False
                self._lose_chance(enemy)
                break
        self.enemies = [item for item in self.enemies if item.alive]

    def _kill_enemy(self, enemy):
        self.score += 1
        x, y = enemy.center
        self.explosions.append(Explosion(x, y))
        self.popups.append(Popup(x, y - 12, "+1", "score"))
        self.events.append({"type": "killed", "score": self.score,
                            "pos": (x, y), "kind": enemy.kind})

    def _lose_chance(self, enemy):
        self.chances -= 1
        self.invincible_timer = self.invincible_seconds
        self.hit_flash = HIT_FLASH_SECONDS
        x, y = enemy.center
        self.explosions.append(Explosion(x, y))
        self.popups.append(Popup(x, y - 12, "机会 -1", "danger"))
        self.events.append({"type": "chance", "chances": self.chances})
        if self.chances <= 0:
            self._finish()

    def _finish(self):
        self.state = self.OVER
        self.armed = False
        self.hold_time = 0.0
        self.last_score = self.score
        data = self.highscores.load()
        self.new_record = self.score > data["best"]
        data["best"] = max(data["best"], self.score)
        data["last"] = self.score
        data["games"] = data.get("games", 0) + 1
        self.highscores.save(data)
        self.record = data
        self.best_score = data["best"]
        self.events.append({"type": "gameover", "score": self.score,
                            "best": self.best_score,
                            "new_record": self.new_record})
