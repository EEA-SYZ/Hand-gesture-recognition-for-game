# -*- coding: utf-8 -*-
"""Environment self-check for the hand-gesture plane game.

Run with the project virtual environment:

    .venv\\Scripts\\python.exe verify_environment.py

It checks the Python runtime, third-party packages, the inference model,
camera availability and the pygame display/font setup.  Exit code 0 means all
critical checks passed.
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
        ("yaml", "PyYAML"),
        ("imutils", "imutils"),
        ("paddle", "paddlepaddle"),
        ("paddlex", "paddlex (compat layer)"),
    ]
    versions = {}
    for module_name, label in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            versions[module_name] = version
            record("import {}".format(label), True, str(version))
        except Exception as exc:
            record("import {}".format(label), False, repr(exc))
    return versions


def check_files():
    required = [
        "plane_cnn.py",
        "platech.ttf",
        "rec_inference_model/__model__",
        "rec_inference_model/__params__",
        "rec_inference_model/model.yml",
        "resources/image/background.png",
        "resources/image/gameover.png",
        "resources/image/shoot.png",
    ]
    for path in required:
        record("file {}".format(path), os.path.exists(path))


def check_model():
    from paddlex.deploy import Predictor

    started = time.time()
    model = Predictor("rec_inference_model", use_gpu=True)
    load_time = time.time() - started
    record("load inference model", True,
           "{}, labels={}, device={}, {:.2f}s".format(
               model.model_name, model.labels,
               "gpu" if model.use_gpu else "cpu", load_time))

    import cv2
    import numpy as np

    # A synthetic skin-coloured blob is enough to exercise the full pipeline
    # (colour conversion, RandomCrop, Normalize, NCHW, softmax).
    image = np.zeros((195, 220, 3), dtype=np.uint8)
    image[:] = (40, 40, 40)
    cv2.ellipse(image, (110, 110), (65, 85), 0, 0, 360, (120, 160, 220), -1)
    cv2.circle(image, (110, 55), 30, (120, 160, 220), -1)

    started = time.time()
    preds = model.predict(image, topk=5)
    infer_time = time.time() - started
    total = sum(item["score"] for item in preds)
    record("run inference", len(preds) == len(model.labels),
           "top1={} ({:.3f}), {:.1f} ms/frame".format(
               preds[0]["category"], preds[0]["score"], infer_time * 1000))
    print("       top-5:", ", ".join(
        "{}={:.3f}".format(p["category"], p["score"]) for p in preds))
    record("softmax probabilities", abs(total - 1.0) < 1e-3,
           "sum={:.6f}".format(total))
    return model


def check_camera():
    import cv2
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
        print("       Run check_camera.bat from a normal desktop session to test the camera.")


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
    print("Hand-gesture plane game - environment check")
    print("=" * 70)
    check_runtime()
    print("-" * 70)
    check_packages()
    print("-" * 70)
    check_files()
    print("-" * 70)
    try:
        check_model()
    except Exception:
        record("load inference model", False, "see traceback below")
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
