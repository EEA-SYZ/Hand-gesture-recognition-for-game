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
        ("imutils", "imutils"),
        ("mediapipe", "mediapipe"),
        ("matplotlib", "matplotlib"),
        ("gesture_mediapipe", "gesture_mediapipe"),
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
        "plane_cnn.py",
        "gesture_mediapipe.py",
        "platech.ttf",
        "resources/image/background.png",
        "resources/image/gameover.png",
        "resources/image/shoot.png",
    ]
    for path in required:
        record("file {}".format(path), os.path.exists(path))


def check_gesture():
    from gesture_mediapipe import GestureRecognizer

    started = time.time()
    recognizer = GestureRecognizer()
    load_time = time.time() - started
    record("load MediaPipe Hands", True, "{:.2f}s".format(load_time))

    import numpy as np

    blank = np.zeros((195, 220, 3), dtype=np.uint8)
    started = time.time()
    label = recognizer.recognize(blank)
    infer_time = time.time() - started
    record("run gesture pipeline", label == "pause",
           "no hand -> {!r}, {:.1f} ms/frame".format(
               label, infer_time * 1000))
    recognizer.close()


def check_camera():
    from camera_compat import open_camera

    camera = open_camera()
    real = getattr(camera, "is_real", True)
    ok, frame = camera.read()
    camera.release()
    shape = None if frame is None else frame.shape
    record("camera source", ok and frame is not None,
           "{} ({})".format("physical" if real else "virtual fallback", shape))
    if not real:
        print("       No usable camera in this process: keyboard fallback is active.")
        print("       Run check_camera.bat as administrator to test the camera.")


def check_pygame():
    import pygame
    from pygame.locals import K_LEFT  # noqa: F401 - import sanity

    pygame.init()
    try:
        screen = pygame.display.set_mode((480, 800))
        record("pygame display", screen is not None,
               "{}x{}".format(*screen.get_size()))
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
        record("load MediaPipe Hands", False, "see traceback below")
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
