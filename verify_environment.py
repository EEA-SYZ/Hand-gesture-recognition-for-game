# -*- coding: utf-8 -*-
"""Environment self-check for the MediaPipe hand-gesture plane game.

Run with the project virtual environment:

    .venv\\Scripts\\python.exe verify_environment.py
"""

from __future__ import absolute_import

import os
import platform
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             " - " + detail if detail else ""))


def check_runtime():
    print("=" * 70)
    print("Python :", sys.version.replace("\n", " "))
    print("Exe    :", sys.executable)
    print("Bits   :", platform.architecture()[0])
    print("System :", platform.platform())
    record("Python >= 3.9", sys.version_info >= (3, 9),
           "{}.{}".format(*sys.version_info[:2]))
    record("64-bit interpreter", platform.architecture()[0] == "64bit")


def check_packages():
    packages = [
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("pygame", "pygame"),
        ("mediapipe", "mediapipe"),
        ("matplotlib", "matplotlib"),
        ("gesture_code", "gesture_code"),
        ("gesture_single", "gesture_single"),
        ("game_core", "game_core"),
        ("plane_game", "plane_game"),
    ]
    for module_name, label in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "local")
            record("import {}".format(label), True, str(version))
        except Exception as exc:
            record("import {}".format(label), False, repr(exc))


def check_files():
    required = [
        "plane_game.py",
        "game_core.py",
        "gesture_code.py",
        "gesture_single.py",
        "platech.ttf",
        "resources/image/background.png",
        "resources/image/gameover.png",
        "resources/image/shoot.png",
    ]
    for path in required:
        record("file {}".format(path), os.path.exists(path))


def check_gesture():
    from gesture_code import direction_of, is_start_pose
    from gesture_single import GestureState, SingleHandRecognizer, roi_rect

    record("mapping P10000 -> left", direction_of("P10000") == "left")
    record("mapping N10000 -> right", direction_of("N10000") == "right")
    record("mapping Xx1111 -> up",
           direction_of("P11111") == "up" and direction_of("N01111") == "up")
    record("mapping X00000 -> down",
           direction_of("P00000") == "down" and
           direction_of("N00000") == "down")
    record("start pose X11111",
           is_start_pose("P11111") and is_start_pose("N11111") and
           not is_start_pose("P01111"))

    x1, y1, x2, y2 = roi_rect((450, 600, 3))
    record("ROI is the top-right quarter at 95%", (x1, y1, x2, y2) ==
           (308, 6, 592, 219), str((x1, y1, x2, y2)))
    record("empty gesture state is pause",
           GestureState().direction == "pause")

    started = time.time()
    recognizer = SingleHandRecognizer()
    load_time = time.time() - started
    record("load MediaPipe Hands", True, "{:.2f}s".format(load_time))

    import numpy as np

    blank = np.zeros((450, 600, 3), dtype=np.uint8)
    started = time.time()
    state = recognizer.process(blank)
    infer_time = time.time() - started
    record("run gesture pipeline", state.direction == "pause",
           "no hand -> {!r}, {:.1f} ms/frame".format(
               state.direction, infer_time * 1000))
    record("the ROI was cut from the frame",
           state.roi == (308, 6, 592, 219), str(state.roi))
    recognizer.close()


def check_camera():
    from camera_compat import open_camera
    from gesture_single import roi_rect

    camera = open_camera()
    real = getattr(camera, "is_real", True)
    ok, frame = camera.read()
    camera.release()
    shape = None if frame is None else frame.shape
    record("camera source", ok and frame is not None,
           "{} ({})".format("physical" if real else "virtual fallback", shape))
    if frame is not None:
        x1, y1, x2, y2 = roi_rect(frame.shape)
        record("ROI inside the camera frame",
               0 <= x1 < x2 <= frame.shape[1] and
               0 <= y1 < y2 <= frame.shape[0],
               str((x1, y1, x2, y2)))
    if not real:
        print("       No usable camera in this process: keyboard fallback is active.")
        print("       Run check_camera.bat as administrator to test the camera.")


def check_pygame():
    import pygame
    from game_core import WINDOW_SIZE

    pygame.init()
    try:
        screen = pygame.display.set_mode(WINDOW_SIZE)
        record("pygame display", screen is not None,
               "{}x{}".format(*screen.get_size()))
        record("window is the 420x700 reference size",
               tuple(screen.get_size()) == (420, 700),
               str(tuple(screen.get_size())))
        font_path = os.path.join(ROOT, "platech.ttf")
        font = pygame.font.Font(font_path, 32)
        surface = font.render("飞机大战", True, (255, 255, 255))
        record("Chinese font rendering", surface.get_width() > 0,
               "{}x{}".format(surface.get_width(), surface.get_height()))
    finally:
        pygame.display.quit()
        pygame.quit()


def main():
    print("MediaPipe hand-gesture plane game - environment check")
    print("=" * 70)
    check_runtime()
    print("-" * 70)
    check_packages()
    print("-" * 70)
    check_files()
    print("-" * 70)
    try:
        check_gesture()
    except Exception:
        record("gesture pipeline", False, "see traceback below")
        traceback.print_exc()
    print("-" * 70)
    try:
        check_camera()
    except Exception:
        record("camera source", False, "see traceback below")
        traceback.print_exc()
    print("-" * 70)
    try:
        check_pygame()
    except Exception:
        record("pygame display", False, "see traceback below")
        traceback.print_exc()

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("=" * 70)
    if failed:
        print("RESULT: {} check(s) failed: {}".format(
            len(failed), ", ".join(failed)))
        return 1
    print("RESULT: all {} checks passed.".format(len(RESULTS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
