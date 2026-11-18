# -*- coding: utf-8 -*-
"""pygame scene and main loop of the hand-gesture plane game.

The game itself (planes, enemies, difficulty, chances) lives in ``game_core``;
this module only loads the artwork, draws the READY / PLAYING / GAME OVER
screens and runs the frame loop.  The layout is designed for the bundled
480x800 artwork and shown in a 420x700 window (the 480x800 canvas scaled to
87.5 %), which is the window size of the reference rhythm game.

Controls
--------
* hold the ``X11111`` hand pose (the whole hand open, thumb included) for
  0.7 s to start a round (no hand, or any other pose, leaves the plane where
  it is);
* ``Xx1111`` up, ``X00000`` down, ``P10000`` left, ``N10000`` right;
* on the game over screen: click 重来 or press space to return to the start
  screen, or hold ``X11111`` for 0.7 s to restart directly;
* the circular hit boxes are outlined on screen: the player's in cyan, the
  enemies' plane-collision circle in red and their larger bullet circle in
  violet;
* arrow keys still move the plane as a keyboard fallback, space starts a round
  when no usable camera is present (keyboard mode);
* ESC quits.
"""

from __future__ import absolute_import

import math
import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from pygame.locals import (KEYDOWN, K_DOWN, K_ESCAPE, K_LEFT, K_RETURN,
                           K_RIGHT, K_SPACE, K_UP, MOUSEBUTTONDOWN, QUIT)

import cv2

from game_core import (BOTTOM_PANEL_HEIGHT, ENEMY_H, HUD_HEIGHT, PLAY_BOTTOM,
                       PLAY_TOP, SCREEN_HEIGHT, SCREEN_WIDTH, WINDOW_SIZE,
                       PlaneGame)
from gesture_single import roi_rect

# ---------------------------------------------------------------------------
# Colours and fonts
# ---------------------------------------------------------------------------
TEXT_LIGHT = (238, 243, 252)
TEXT_DIM = (150, 160, 180)
ACCENT = (110, 200, 255)
ACCENT_DIM = (70, 120, 170)
GOLD = (255, 205, 90)
WARN = (255, 170, 90)
DANGER = (255, 92, 110)
RING = (255, 214, 64)
PANEL = (14, 18, 30)
PANEL_BORDER = (52, 64, 92)
BUTTON = (70, 165, 118)
BUTTON_HOVER = (92, 204, 144)

#: Colours of the circular hit boxes drawn over the planes.
HITBOX_PLAYER = (0, 190, 255)
HITBOX_ENEMY = (255, 70, 70)
HITBOX_BULLET = (200, 120, 255)

CAMERA_WINDOW = "Camera - Plane Game"

#: Restart button of the game over screen (canvas coordinates).
RESTART_BUTTON = pygame.Rect(120, 428, 240, 58)

#: Chinese labels of the four directions.
DIRECTION_LABELS = {"up": "上", "down": "下", "left": "左", "right": "右",
                    "pause": "静止"}

#: Sprite rectangles inside ``resources/image/shoot.png``.
PLAYER_RECT = (0, 99, 102, 126)
ENEMY_RECT = (534, 612, 57, 43)
ENEMY_DOWN_RECT = (267, 347, 57, 43)
BULLET_RECT = (1004, 987, 9, 21)


def load_fonts(root):
    path = os.path.join(root, "platech.ttf")
    return {
        "title": pygame.font.Font(path, 64),
        "sub": pygame.font.Font(path, 20),
        "huge": pygame.font.Font(path, 52),
        "big": pygame.font.Font(path, 40),
        "medium": pygame.font.Font(path, 28),
        "small": pygame.font.Font(path, 22),
        "tiny": pygame.font.Font(path, 18),
        "score": pygame.font.Font(path, 34),
    }


def draw_text(surface, font, text, color, center=None, topleft=None,
              topright=None):
    rendered = font.render(text, True, color)
    rect = rendered.get_rect()
    if center is not None:
        rect.center = center
    elif topleft is not None:
        rect.topleft = topleft
    elif topright is not None:
        rect.topright = topright
    surface.blit(rendered, rect)
    return rect


def draw_text_alpha(surface, font, text, color, center, alpha):
    rendered = font.render(text, True, color)
    rendered.set_alpha(int(max(0, min(255, alpha))))
    surface.blit(rendered, rendered.get_rect(center=center))


def draw_progress_bar(surface, rect, progress, color, back=(40, 48, 66)):
    bar = pygame.Rect(rect)
    pygame.draw.rect(surface, back, bar, border_radius=max(2, bar.h // 2))
    width = int(bar.w * max(0.0, min(1.0, progress)))
    if width > 2:
        fill = pygame.Rect(bar.x, bar.y, width, bar.h)
        pygame.draw.rect(surface, color, fill,
                         border_radius=max(2, bar.h // 2))
    pygame.draw.rect(surface, (70, 82, 108), bar, 1,
                     border_radius=max(2, bar.h // 2))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
class PlaneScene(object):
    """Draws the READY / PLAYING / GAME OVER screens."""

    def __init__(self, root=None, size=(SCREEN_WIDTH, SCREEN_HEIGHT)):
        self.root = root or os.path.dirname(os.path.abspath(__file__))
        self.size = size
        self.fonts = load_fonts(self.root)
        self.mouse_pos = (0, 0)
        self.background = self._load_background()
        self._load_sprites()

    # -- assets ---------------------------------------------------------
    def _load_background(self):
        path = os.path.join(self.root, "resources", "image", "background.png")
        image = pygame.image.load(path)
        if image.get_size() != self.size:
            image = pygame.transform.smoothscale(image, self.size)
        return image.convert()

    def _load_sprites(self):
        path = os.path.join(self.root, "resources", "image", "shoot.png")
        sheet = pygame.image.load(path).convert_alpha()
        self.player_img = sheet.subsurface(pygame.Rect(*PLAYER_RECT)).copy()
        self.enemy_img = sheet.subsurface(pygame.Rect(*ENEMY_RECT)).copy()
        self.enemy_down_img = sheet.subsurface(
            pygame.Rect(*ENEMY_DOWN_RECT)).copy()
        self.bullet_img = sheet.subsurface(pygame.Rect(*BULLET_RECT)).copy()
        # The enemy sprite points down: rotate it 90 degrees clockwise for a
        # plane flying to the right and counter-clockwise for one going left.
        self.enemy_right_img = pygame.transform.rotate(self.enemy_img, 90)
        self.enemy_left_img = pygame.transform.rotate(self.enemy_img, -90)
        self.player_icon = pygame.transform.smoothscale(self.player_img,
                                                        (26, 32))
        self.player_icon_dim = self.player_icon.copy()
        self.player_icon_dim.fill((62, 66, 82, 255),
                                  special_flags=pygame.BLEND_RGBA_MULT)
        self.player_badge = pygame.transform.smoothscale(self.player_img,
                                                         (84, 104))

    # -- entry point ----------------------------------------------------
    def draw(self, canvas, game, keyboard_mode=False, warning=None):
        canvas.blit(self.background, (0, 0))
        if game.state == PlaneGame.PLAYING:
            self._draw_playfield(canvas, game)
            self._draw_hud(canvas, game)
            self._draw_bottom(canvas, game, keyboard_mode)
            return
        if game.state == PlaneGame.OVER:
            # Keep the frozen battlefield visible behind the panel.
            self._draw_playfield(canvas, game)
            self._draw_dim(canvas, 170)
            self._draw_over(canvas, game, keyboard_mode)
            return
        self._draw_dim(canvas, 190)
        self._draw_ready(canvas, game, keyboard_mode, warning)

    # -- READY ----------------------------------------------------------
    def _draw_ready(self, canvas, game, keyboard_mode, warning):
        fonts = self.fonts
        draw_text(canvas, fonts["title"], "飞机大战", TEXT_LIGHT,
                  center=(SCREEN_WIDTH // 2, 116))
        draw_text(canvas, fonts["sub"], "GESTURE  PLANE  WAR", ACCENT_DIM,
                  center=(SCREEN_WIDTH // 2, 162))

        card = pygame.Rect(56, 194, 368, 212)
        pygame.draw.rect(canvas, PANEL, card, border_radius=18)
        pygame.draw.rect(canvas, PANEL_BORDER, card, 2, border_radius=18)
        draw_text(canvas, fonts["tiny"], "开始手型", TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 218))
        draw_text(canvas, fonts["huge"], "X11111", TEXT_LIGHT,
                  center=(SCREEN_WIDTH // 2, 262))
        draw_text(canvas, fonts["tiny"], "五指全部张开，拇指也伸出", TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 302))
        draw_text(canvas, fonts["small"], "保持 0.7 秒开始游戏", ACCENT,
                  center=(SCREEN_WIDTH // 2, 332))
        draw_progress_bar(canvas, pygame.Rect(110, 356, 260, 14),
                          game.hold_progress, ACCENT)

        draw_text(canvas, fonts["small"],
                  "历史最高  {}".format(game.best_score), TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 432))

        draw_text(canvas, fonts["tiny"], "手势对照", TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 472))
        draw_text(canvas, fonts["small"], "Xx1111 上      X00000 下", TEXT_LIGHT,
                  center=(SCREEN_WIDTH // 2, 502))
        draw_text(canvas, fonts["small"], "P10000 左      N10000 右", TEXT_LIGHT,
                  center=(SCREEN_WIDTH // 2, 532))
        draw_text(canvas, fonts["tiny"], "其他手型：飞机保持不动", TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 562))

        if keyboard_mode:
            draw_text(canvas, fonts["tiny"],
                      "键盘模式：方向键移动，空格开始 / 重来", WARN,
                      center=(SCREEN_WIDTH // 2, 604))
        elif warning:
            draw_text(canvas, fonts["tiny"], warning, WARN,
                      center=(SCREEN_WIDTH // 2, 604))
        draw_text(canvas, fonts["tiny"], "把一只手放进画面右上角的绿框", TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 632))
        draw_text(canvas, fonts["tiny"], "ESC 退出", TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 658))

        badge_rect = self.player_badge.get_rect(
            center=(SCREEN_WIDTH // 2, 714))
        canvas.blit(self.player_badge, badge_rect)

    # -- PLAYING --------------------------------------------------------
    def _draw_playfield(self, canvas, game):
        self._draw_warnings(canvas, game)

        for enemy in game.enemies:
            if enemy.kind == "top":
                image = self.enemy_img
            elif enemy.moving_right:
                image = self.enemy_right_img
            else:
                image = self.enemy_left_img
            rect = image.get_rect(center=(int(round(enemy.center[0])),
                                          int(round(enemy.center[1]))))
            canvas.blit(image, rect)

        for bullet in game.bullets:
            canvas.blit(self.bullet_img, bullet.rect)

        self._draw_player(canvas, game)

        for explosion in game.explosions:
            alpha = int(255 * (1.0 - explosion.progress))
            image = self.enemy_down_img.copy()
            image.set_alpha(alpha)
            rect = image.get_rect(center=(int(round(explosion.x)),
                                          int(round(explosion.y))))
            canvas.blit(image, rect)

        for popup in game.popups:
            progress = popup.age / popup.ttl if popup.ttl > 0 else 1.0
            color = GOLD if popup.kind == "score" else DANGER
            draw_text_alpha(canvas, self.fonts["small"], popup.text,
                            color, (popup.x, popup.y - 24 * progress),
                            255 * (1.0 - progress))

        self._draw_hitboxes(canvas, game)

        if game.hit_flash > 0.0:
            self._draw_border(canvas, DANGER, game.hit_flash / 0.5)

    def _draw_hitboxes(self, canvas, game):
        """Outline the hit boxes: the player's and both circles of every enemy."""
        for enemy in game.enemies:
            center = (int(round(enemy.center[0])),
                      int(round(enemy.center[1])))
            # The larger bullet circle (thin) first, the collision circle on
            # top (bold).
            pygame.draw.circle(canvas, HITBOX_BULLET, center,
                               int(round(enemy.bullet_radius)), 1)
            pygame.draw.circle(canvas, HITBOX_ENEMY, center,
                               int(round(enemy.hit_radius)), 2)
        center = (int(round(game.player.center[0])),
                  int(round(game.player.center[1])))
        pygame.draw.circle(canvas, HITBOX_PLAYER, center,
                           int(round(game.player.radius)), 2)

    def _draw_player(self, canvas, game):
        player = game.player
        visible = True
        if game.state == PlaneGame.PLAYING and game.invincible:
            # Blink at roughly 6 Hz while the invincibility lasts.
            visible = int(game.time * 12) % 2 == 0
        if visible:
            canvas.blit(self.player_img, player.rect)
        if game.state == PlaneGame.PLAYING and game.invincible:
            self._draw_invincible_ring(canvas, game)

    def _draw_invincible_ring(self, canvas, game):
        """Yellow ring that depletes like a loading spinner around the plane."""
        center_x, center_y = game.player.center
        radius = max(game.player.w, game.player.h) / 2.0 + 14.0
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.draw.circle(overlay, (RING[0], RING[1], RING[2], 45),
                           (int(center_x), int(center_y)), int(radius), 7)
        fraction = game.invincible_progress
        if fraction > 0.002:
            arc_rect = pygame.Rect(0, 0, int(radius * 2), int(radius * 2))
            arc_rect.center = (int(center_x), int(center_y))
            start = -math.pi / 2.0
            stop = start + 2.0 * math.pi * fraction
            pygame.draw.arc(overlay, (RING[0], RING[1], RING[2], 235),
                            arc_rect, start, stop, 6)
        canvas.blit(overlay, (0, 0))

    def _draw_warnings(self, canvas, game):
        """Countdown markers of the upcoming side enemies."""
        for warning in game.warnings:
            center_y = int(round(warning.y + ENEMY_H / 2.0))
            pulse = 0.55 + 0.45 * abs(math.sin(game.time * 7.0))
            color = (255, int(140 * pulse) + 55, int(50 * pulse) + 30)
            # Dashed track showing the row the enemy will fly along.
            for x in range(0, SCREEN_WIDTH, 20):
                pygame.draw.line(canvas, color, (x, center_y),
                                 (x + 10, center_y), 2)
            size = 13
            if warning.side == "L":
                edge_x = 6
                arrow = [(edge_x, center_y - size), (edge_x, center_y + size),
                         (edge_x + size + 4, center_y)]
                text_x = edge_x + size + 18
            else:
                edge_x = SCREEN_WIDTH - 6
                arrow = [(edge_x, center_y - size), (edge_x, center_y + size),
                         (edge_x - size - 4, center_y)]
                text_x = edge_x - size - 18
            pygame.draw.polygon(canvas, color, arrow)
            draw_text(canvas, self.fonts["medium"],
                      str(warning.remaining), WARN,
                      center=(text_x, center_y - 28))
            draw_text(canvas, self.fonts["tiny"], "敌机", WARN,
                      center=(text_x, center_y + 24))

    def _draw_hud(self, canvas, game):
        fonts = self.fonts
        strip = pygame.Surface((SCREEN_WIDTH, HUD_HEIGHT), pygame.SRCALPHA)
        strip.fill((9, 13, 22, 214))
        canvas.blit(strip, (0, 0))
        pygame.draw.line(canvas, (34, 44, 66), (0, HUD_HEIGHT - 1),
                         (SCREEN_WIDTH, HUD_HEIGHT - 1), 2)

        draw_text(canvas, fonts["tiny"], "击落", TEXT_DIM, topleft=(16, 8))
        draw_text(canvas, fonts["score"], str(game.score), TEXT_LIGHT,
                  topleft=(16, 26))
        draw_text(canvas, fonts["tiny"], "最高 {}".format(game.best_score),
                  TEXT_DIM, topleft=(16, 62))

        draw_text(canvas, fonts["tiny"],
                  "阶段 {} · {}".format(game.stage,
                                        game.difficulty.label), TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 16))
        draw_progress_bar(canvas,
                          pygame.Rect(SCREEN_WIDTH // 2 - 90, 34, 180, 10),
                          game.difficulty_value, (255, 190, 90))
        draw_text(canvas, fonts["tiny"],
                  "存活 {:.1f}s".format(game.elapsed), TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, 60))

        for index in range(game.max_chances):
            icon = (self.player_icon if index < game.chances
                    else self.player_icon_dim)
            center = (SCREEN_WIDTH - 28 - index * 32, 26)
            canvas.blit(icon, icon.get_rect(center=center))
        draw_text(canvas, fonts["tiny"], "机会", TEXT_DIM,
                  topright=(SCREEN_WIDTH - 16, 54))

    def _draw_bottom(self, canvas, game, keyboard_mode):
        fonts = self.fonts
        panel = pygame.Surface((SCREEN_WIDTH, BOTTOM_PANEL_HEIGHT),
                               pygame.SRCALPHA)
        panel.fill((9, 13, 22, 214))
        canvas.blit(panel, (0, PLAY_BOTTOM))
        pygame.draw.line(canvas, (34, 44, 66), (0, PLAY_BOTTOM),
                         (SCREEN_WIDTH, PLAY_BOTTOM), 2)

        if keyboard_mode:
            text = "键盘模式：方向键移动"
            color = WARN
        else:
            code = game.last_code or "--"
            label = DIRECTION_LABELS.get(game.last_direction, "静止")
            text = "手势 {}  →  {}".format(code, label)
            color = TEXT_LIGHT
        draw_text(canvas, fonts["medium"], text, color,
                  topleft=(16, PLAY_BOTTOM + 14))

        if game.invincible:
            draw_text(canvas, fonts["small"], "无敌 {:.1f}s".format(
                game.invincible_timer), RING,
                topright=(SCREEN_WIDTH - 16, PLAY_BOTTOM + 18))
        else:
            draw_text(canvas, fonts["tiny"], "右上角绿框", TEXT_DIM,
                      topright=(SCREEN_WIDTH - 16, PLAY_BOTTOM + 22))

    # -- GAME OVER ------------------------------------------------------
    def _draw_over(self, canvas, game, keyboard_mode):
        fonts = self.fonts
        panel = pygame.Rect(44, 176, 392, 412)
        pygame.draw.rect(canvas, PANEL, panel, border_radius=18)
        pygame.draw.rect(canvas, PANEL_BORDER, panel, 2, border_radius=18)

        draw_text(canvas, fonts["big"], "游戏结束", TEXT_LIGHT,
                  center=(SCREEN_WIDTH // 2, panel.y + 48))
        draw_text(canvas, fonts["tiny"], "本局击落", TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, panel.y + 96))
        draw_text(canvas, fonts["huge"], str(game.score), TEXT_LIGHT,
                  center=(SCREEN_WIDTH // 2, panel.y + 136))
        draw_text(canvas, fonts["small"],
                  "历史最高  {}".format(game.best_score), TEXT_DIM,
                  center=(SCREEN_WIDTH // 2, panel.y + 190))
        if game.new_record:
            draw_text(canvas, fonts["small"], "新纪录！", GOLD,
                      center=(SCREEN_WIDTH // 2, panel.y + 222))
        else:
            draw_text(canvas, fonts["tiny"],
                      "存活 {:.1f} 秒".format(game.survived), TEXT_DIM,
                      center=(SCREEN_WIDTH // 2, panel.y + 222))

        hovered = RESTART_BUTTON.collidepoint(self.mouse_pos)
        pygame.draw.rect(canvas, BUTTON_HOVER if hovered else BUTTON,
                         RESTART_BUTTON, border_radius=14)
        draw_text(canvas, fonts["medium"], "重来", (12, 24, 18),
                  center=RESTART_BUTTON.center)

        draw_text(canvas, fonts["tiny"], "点击重来 / 空格 → 开始界面",
                  TEXT_DIM, center=(SCREEN_WIDTH // 2, panel.y + 326))
        draw_text(canvas, fonts["tiny"], "保持 X11111 0.7 秒直接重来",
                  ACCENT_DIM, center=(SCREEN_WIDTH // 2, panel.y + 352))
        if keyboard_mode:
            draw_text(canvas, fonts["tiny"], "键盘模式：空格直接重来", WARN,
                      center=(SCREEN_WIDTH // 2, panel.y + 378))

    # -- effects --------------------------------------------------------
    def _draw_dim(self, canvas, alpha):
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        overlay.fill((4, 6, 12, int(alpha)))
        canvas.blit(overlay, (0, 0))

    def _draw_border(self, canvas, color, strength):
        alpha = int(max(0, min(255, 170 * strength)))
        overlay = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.draw.rect(overlay, (color[0], color[1], color[2], alpha),
                         pygame.Rect(0, 0, SCREEN_WIDTH,
                                     SCREEN_HEIGHT).inflate(-14, -14), 10,
                         border_radius=12)
        canvas.blit(overlay, (0, 0))


# ---------------------------------------------------------------------------
# Camera preview overlay (ASCII only: OpenCV cannot render CJK here)
# ---------------------------------------------------------------------------
def annotate_camera(image, game, keyboard_mode=False):
    """Draw the game status and the recognition result on the preview."""
    frame = image.copy()
    height, width = frame.shape[:2]
    scale = max(0.75, min(1.6, width / 640.0))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Recognition area (top-right quarter, 95 % of it).
    x1, y1, x2, y2 = roi_rect(frame.shape)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "ROI top-right 1/4 x95%",
                (x1, min(height - 4, y2 + int(16 * scale))), font,
                0.44 * scale, (0, 255, 0), 1, cv2.LINE_AA)

    # Bottom status strip (kept away from the ROI, which is in the top half).
    strip = int(64 * scale)
    cv2.rectangle(frame, (0, height - strip), (width, height),
                  (18, 20, 28), -1)
    states = {
        PlaneGame.READY: "START: hold X11111 for 0.7 s",
        PlaneGame.PLAYING: "PLAYING",
        PlaneGame.OVER: "GAME OVER - restart on the game window",
    }
    cv2.putText(frame, states.get(game.state, ""),
                (int(10 * scale), height - int(46 * scale)), font,
                0.46 * scale, (168, 176, 194), 1, cv2.LINE_AA)
    cv2.putText(frame, "SCORE {}   BEST {}   CHANCES {}   STAGE {}".format(
        game.score, game.best_score, max(0, game.chances), game.stage),
        (int(10 * scale), height - int(30 * scale)), font, 0.52 * scale,
        (235, 240, 250), 1, cv2.LINE_AA)

    if keyboard_mode:
        hand = "KEYBOARD MODE: arrow keys, SPACE to start"
        color = (255, 190, 90)
    else:
        hand = "HAND {}  ->  {}".format(game.last_code or "--",
                                        game.last_direction.upper())
        color = (150, 210, 255)
    cv2.putText(frame, hand, (int(10 * scale), height - int(12 * scale)),
                font, 0.52 * scale, color, 1, cv2.LINE_AA)

    if game.state in (PlaneGame.READY, PlaneGame.OVER):
        bar_w, bar_h = int(190 * scale), int(8 * scale)
        bar_x = width - bar_w - int(12 * scale)
        bar_y = height - int(26 * scale)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (60, 60, 60), -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * game.hold_progress), bar_y + bar_h),
                      (0, 220, 120), -1)
    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def _keyboard_direction():
    """Direction requested by the arrow keys, or ``None``."""
    keys = pygame.key.get_pressed()
    if keys[K_LEFT]:
        return "left"
    if keys[K_RIGHT]:
        return "right"
    if keys[K_UP]:
        return "up"
    if keys[K_DOWN]:
        return "down"
    return None


def _start_or_restart(game, keyboard_mode):
    """Handle space / the restart button.

    With a camera the round always needs the ``X11111`` pose, so space and the
    button only go back to the start screen.  Without a usable camera the pose
    can never be recognised, hence keyboard mode starts the round directly.
    """
    if game.state == PlaneGame.OVER:
        if keyboard_mode:
            game.start()
        else:
            game.restart()
    elif game.state == PlaneGame.READY and keyboard_mode:
        game.start()


def run(source, scene=None, hook=None, fps=60, keyboard_mode=False,
        show_camera=True, game=None, warning=None,
        camera_window=CAMERA_WINDOW,
        window_title="手势打飞机 / Gesture Plane War", fixed_dt=None):
    """Run the game loop until the user quits.

    Args:
        source: object with ``latest() -> (image, GestureState)``.  The real
            game passes a :class:`gesture_single.CameraWorker`; the screenshot
            capture passes a scripted source with ``image = None``.
        hook: optional ``hook(frame_index, game, source)`` called after every
            rendered frame; return ``False`` to stop (used by capture_game).
        game: optional pre-built :class:`game_core.PlaneGame` (tests inject a
            temporary high score file).
        fixed_dt: use this constant step instead of the measured frame time
            (the screenshot capture uses it to make the pictures reproducible).
    """
    window = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption(window_title)
    canvas = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    scaled = pygame.Surface(WINDOW_SIZE)
    scene = scene or PlaneScene()
    game = game or PlaneGame()
    clock = pygame.time.Clock()

    def to_scene(pos):
        """Window pixel -> canvas pixel (the canvas is scaled down on show)."""
        return (pos[0] * SCREEN_WIDTH / float(WINDOW_SIZE[0]),
                pos[1] * SCREEN_HEIGHT / float(WINDOW_SIZE[1]))

    camera_visible = show_camera and source is not None
    running = True
    frame_index = 0
    try:
        while running:
            frame_ms = clock.tick(fps)
            dt = (min(frame_ms / 1000.0, 0.05) if fixed_dt is None
                  else float(fixed_dt))
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        running = False
                    elif event.key in (K_SPACE, K_RETURN):
                        _start_or_restart(game, keyboard_mode)
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    if (game.state == PlaneGame.OVER and
                            RESTART_BUTTON.collidepoint(to_scene(event.pos))):
                        _start_or_restart(game, keyboard_mode)

            if source is not None:
                image, gesture = source.latest()
            else:
                image, gesture = None, None
            game.update(dt, gesture, _keyboard_direction())

            scene.mouse_pos = to_scene(pygame.mouse.get_pos())
            scene.draw(canvas, game, keyboard_mode, warning)
            pygame.transform.smoothscale(canvas, WINDOW_SIZE, scaled)
            window.blit(scaled, (0, 0))
            pygame.display.update()

            if image is not None and camera_visible:
                try:
                    cv2.imshow(camera_window,
                               annotate_camera(image, game, keyboard_mode))
                    cv2.waitKey(1)
                except Exception:
                    # The preview window was closed or the GUI is unavailable;
                    # the game itself keeps running.
                    camera_visible = False

            frame_index += 1
            if hook is not None and hook(frame_index, game, source) is False:
                running = False
    finally:
        pygame.display.quit()
        if camera_visible:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
    return game


def main(argv=None):
    """Open the camera, start the recogniser thread and run the game."""
    pygame.init()

    from camera_compat import open_camera
    from gesture_single import CameraWorker, SingleHandRecognizer

    camera = None
    recognizer = None
    worker = None
    warning = None
    keyboard_mode = False
    try:
        camera = open_camera()
        if not getattr(camera, "is_real", True):
            keyboard_mode = True
            if getattr(camera, "reason", "") == "forced":
                warning = "摄像头已禁用（GAME_CAMERA=virtual）：键盘模式"
            else:
                warning = "摄像头不可用：键盘模式，可运行 check_camera.bat 检查"
        if not keyboard_mode:
            recognizer = SingleHandRecognizer()
        # A real camera is mirrored like a selfie; the virtual fallback is not.
        worker = CameraWorker(camera, recognizer,
                              mirror=getattr(camera, "is_real", True)).start()
        return run(worker, keyboard_mode=keyboard_mode, warning=warning)
    finally:
        if worker is not None:
            worker.close()
        elif camera is not None:
            try:
                camera.release()
            except Exception:
                pass
        if recognizer is not None:
            recognizer.close()
        pygame.quit()


if __name__ == "__main__":
    main(sys.argv[1:])
