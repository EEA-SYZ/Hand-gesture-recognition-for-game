# -*- coding: utf-8 -*-
"""MediaPipe Hands based gesture recognition for the plane game.

Recognition runs on the green ROI of the camera frame (one hand only).  The
skeleton (21 landmarks + connections) is drawn directly on that ROI so the
OpenCV preview shows what MediaPipe sees.

Gesture mapping:
    open palm                      -> up
    fist                           -> down
    thumbs-up pointing left        -> left
    thumbs-up pointing right       -> right
    anything else / no hand        -> pause

Finger state is determined by the joint angle at the PIP node: a straight
finger is close to 180 degrees, while a folded finger curls back toward the
palm and its PIP angle drops below ``FOLD_ANGLE``.
"""

from __future__ import absolute_import

import collections
import os

# Keep MediaPipe/TFLite from flooding the game console.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np

# Joint-angle thresholds used to decide whether a finger is folded.
FOLD_ANGLE = 150.0         # PIP angle below this means the finger is folded
THUMB_OPEN_ANGLE = 150.0   # thumb IP angle above this means it is extended
THUMB_DIST_RATIO = 0.75    # thumb tip distance / palm size (avoid thumb-on-palm)
HORIZONTAL_RATIO = 0.80    # |dx| / |dy| for a sideways thumb
HORIZONTAL_MIN = 0.25      # |dx| / palm size for a sideways thumb

# Landmark indices.
WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
PALM_POINTS = (0, 5, 9, 13, 17)
FINGERS = (
    (5, 6, 8),     # index: MCP, PIP, TIP
    (9, 10, 12),   # middle
    (13, 14, 16),  # ring
    (17, 18, 20),  # pinky
)


def _as_xy(landmarks):
    return np.array([[float(lm.x), float(lm.y)] for lm in landmarks],
                    dtype=np.float32)


def _joint_angle(a, b, c):
    """Angle ABC in degrees (2D projection of the landmarks)."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_value = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def classify_landmarks(landmarks):
    """Classify one MediaPipe hand into up/down/left/right/pause.

    Args:
        landmarks: sequence of 21 MediaPipe landmarks with ``x``/``y``.
    """
    pts = _as_xy(landmarks)
    if pts.shape != (21, 2):
        return "pause"

    palm = pts[list(PALM_POINTS)].mean(axis=0)
    palm_size = float(np.linalg.norm(pts[WRIST] - pts[9]))
    if palm_size < 1e-6:
        return "pause"

    open_count = 0
    for mcp, pip, tip in FINGERS:
        # A straight finger has a PIP angle close to 180 degrees; a folded
        # finger's tip curls back toward the palm and the angle drops.
        pip_angle = _joint_angle(pts[mcp], pts[pip], pts[tip])
        if pip_angle > FOLD_ANGLE:
            open_count += 1

    thumb_angle = _joint_angle(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP])
    thumb_dist = float(np.linalg.norm(pts[THUMB_TIP] - palm))
    thumb_open = (thumb_angle > THUMB_OPEN_ANGLE and
                  thumb_dist > palm_size * THUMB_DIST_RATIO)

    thumb_vec = pts[THUMB_TIP] - pts[THUMB_MCP]
    dx, dy = float(thumb_vec[0]), float(thumb_vec[1])
    sideways = (abs(dx) > abs(dy) * HORIZONTAL_RATIO and
                abs(dx) > palm_size * HORIZONTAL_MIN)

    # Thumbs-up held sideways: only the thumb is out.
    if thumb_open and open_count == 0 and sideways:
        return "left" if dx < 0 else "right"

    # Open palm: all four long fingers are extended.
    if open_count >= 3:
        return "up"

    # Fist: four long fingers curled and the thumb is not sticking out.
    if open_count == 0 and not thumb_open:
        return "down"

    # Everything else (vertical thumbs-up, pointing, peace, ...) is idle.
    return "pause"


class GestureRecognizer(object):
    """Thin wrapper around MediaPipe Hands for one hand in a BGR ROI."""

    def __init__(self,
                 min_detection_confidence=0.60,
                 min_tracking_confidence=0.50,
                 model_complexity=1,
                 smooth_window=5):
        self._hands_module = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._hands = self._hands_module.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._history = collections.deque(maxlen=smooth_window)
        self._last = "pause"

    def _stable_label(self, label):
        self._history.append(label)
        if len(self._history) < 3:
            return label
        counts = collections.Counter(self._history)
        most_common, count = counts.most_common(1)[0]
        if count >= 2:
            return most_common
        return self._last

    def recognize(self, roi_bgr):
        """Return gesture label and draw skeleton on ``roi_bgr`` in place."""
        if roi_bgr is None or roi_bgr.size == 0:
            return "pause"

        rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        label = "pause"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            label = classify_landmarks(hand_landmarks.landmark)
            try:
                self._drawing.draw_landmarks(
                    roi_bgr,
                    hand_landmarks,
                    self._hands_module.HAND_CONNECTIONS,
                    self._drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3),
                    self._drawing.DrawingSpec(
                        color=(0, 0, 255), thickness=2))
                cv2.putText(roi_bgr, label, (8, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 255), 2, cv2.LINE_AA)
            except Exception:
                # Drawing must never break gesture control.
                pass

        stable = self._stable_label(label)
        self._last = stable
        return stable

    def close(self):
        self._hands.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
