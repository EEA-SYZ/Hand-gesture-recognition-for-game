# -*- coding: utf-8 -*-
"""Camera source helpers for the hand-gesture game.

The game was written for a machine with a webcam.  This module keeps the
original ``cv2.VideoCapture`` interface but adds two things:

* a more patient device probe (DirectShow -> Media Foundation -> default),
* a *virtual* camera fallback used when no camera is present, so the game can
  still start and be played with the arrow keys instead of crashing.

Set the ``GAME_CAMERA`` environment variable to choose another device index or
a video file, e.g. ``set GAME_CAMERA=1`` or ``set GAME_CAMERA=demo.mp4``.
"""

from __future__ import absolute_import

import os
import time

import cv2
import numpy as np

try:  # Keep the console readable; we print our own diagnostics.
    cv2.setLogLevel(0)
except Exception:
    pass

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 450
# Same gesture box as plane_cnn.py: (x1, y1) - (x2, y2)
ROI_RECT = (360, 90, 580, 285)


class VirtualCamera(object):
    """A tiny stand-in for ``cv2.VideoCapture`` when no camera is available."""

    is_real = False

    def __init__(self, width=WINDOW_WIDTH, height=WINDOW_HEIGHT,
                 reason="unavailable"):
        self.width = width
        self.height = height
        self.reason = reason  # "unavailable" or "forced"
        self._start = time.time()
        self._base = self._make_background()

    def _make_background(self):
        # Vertical dark-blue gradient.
        top = np.array([28, 18, 12], dtype=np.float32)
        bottom = np.array([70, 35, 20], dtype=np.float32)
        rows = np.linspace(0.0, 1.0, self.height, dtype=np.float32)[:, None]
        gradient = top[None, :] * (1.0 - rows) + bottom[None, :] * rows
        frame = np.repeat(gradient[:, None, :], self.width, axis=1)
        return frame.astype(np.uint8)

    def isOpened(self):  # noqa: N802 - mimics cv2.VideoCapture
        return True

    def read(self):
        frame = self._base.copy()
        x1, y1, x2, y2 = ROI_RECT
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "CAMERA UNAVAILABLE", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, "Play with arrow keys", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, "Gesture box (simulated)", (x1, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1,
                    cv2.LINE_AA)
        # Moving scan line makes it obvious the preview is alive.
        y = int((time.time() - self._start) * 80) % self.height
        cv2.line(frame, (0, y), (self.width, y), (60, 60, 60), 1)
        return True, frame

    def release(self):
        pass

    def set(self, prop, value):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        return 0.0


def _probe(index):
    for name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
        backend = getattr(cv2, name, None)
        if backend is None:
            continue
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # Some backends open lazily; retry a few frames.
            for _ in range(10):
                grabbed, frame = cap.read()
                if grabbed and frame is not None and frame.size:
                    return cap, name
                time.sleep(0.05)
            cap.release()
        except Exception:
            if cap is not None:
                cap.release()
    return None, None


def open_camera(source=None):
    """Open a real camera, a video file, or the virtual fallback.

    Args:
        source: device index, video path, ``"auto"``, or ``"virtual"``.
            ``None`` reads the ``GAME_CAMERA`` environment variable.  When
            unset, indices 0-5 are scanned with DirectShow, Media Foundation
            and the default backend, so a camera does not have to be index 0.

    Returns:
        An object exposing ``read()``, ``release()`` and ``isOpened()``.
        Real ``cv2.VideoCapture`` objects have no ``is_real`` attribute; use
        ``getattr(camera, "is_real", True)`` to tell them apart.  The virtual
        fallback has ``reason`` set to ``"unavailable"`` or ``"forced"``.
    """
    if source is None:
        source = os.environ.get("GAME_CAMERA", "auto")
    source = str(source).strip() or "auto"

    if source.lower() in ("virtual", "keyboard", "none"):
        print("[INFO] Keyboard mode requested (GAME_CAMERA={}).".format(source))
        return VirtualCamera(reason="forced")

    if source.lower() == "auto":
        print("[INFO] Scanning camera indices 0-5 ...")
        indices = range(6)
    elif source.isdigit():
        indices = [int(source)]
    else:
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            print("[INFO] Video source opened: {}".format(source))
            return cap
        cap.release()
        print("[WARN] Video source cannot be opened: {}".format(source))
        print("[WARN] Falling back to virtual camera.")
        return VirtualCamera(reason="unavailable")

    for index in indices:
        cap, backend = _probe(index)
        if cap is not None:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            print("[INFO] Camera {} opened via {} ({}x{})".format(
                index, backend, width, height))
            return cap

    if source.lower() == "auto":
        print("[WARN] No camera could be opened on indices 0-5 "
              "(DirectShow / Media Foundation / AUTO).")
    else:
        print("[WARN] Camera {} could not be opened.".format(source))
    print("[WARN] If a camera is present: close other camera apps "
          "(WeChat / Teams / Edge), enable 'Settings > Privacy & security > "
          "Camera', and run check_camera.bat. Falling back to keyboard mode.")
    return VirtualCamera(reason="unavailable")
