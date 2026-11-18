# -*- coding: utf-8 -*-
"""Single-hand MediaPipe recognition for the plane game.

The hand shape / thumb orientation maths is a direct port of the two-hand
rhythm game (``Hand-gesture-ps-game/gesture_twohands.py``) so both projects
judge a hand with (almost) the same code.  Only the parts that are specific to
"one hand drives a plane" differ:

* ``max_num_hands=1``: exactly one hand is recognised and it steers the plane;
* the hand is treated like the **right** hand of the two-hand game, which is the
  example given in the task: ``P`` = the thumb points left, ``N`` = it points
  right.  With that convention ``P10000`` means "left" and ``N10000`` "right";
* recognition runs on the **top-right quarter** of the (mirrored) camera frame,
  shrunk to 95 % of that quarter - see :func:`roi_rect`.

Everything else is identical to the reference implementation:

* a long finger is extended when its PIP and DIP joint angles are almost
  straight (measured in 3D, so a finger pointing at the camera is not judged
  from a shortened 2D projection);
* the thumb is extended when it is not flexed towards the palm and its tip is
  far enough from the palm centre;
* a per-character majority vote over the last few frames removes jitter;
* the 21 landmarks and their connections are drawn on the ROI, which is a view
  into the camera frame, so the skeleton shows up in the preview window.

The plane game mapping (see ``gesture_code``)::

    Xx1111 -> up      X00000 -> down
    P10000 -> left    N10000 -> right      every other pose -> pause

``X11111`` (the whole hand open, thumb included) held for 0.7 s starts and
restarts a round; it is reported through :attr:`GestureState.start_pose`.
"""

from __future__ import absolute_import

import collections
import os
import threading
import time
import traceback

# Keep MediaPipe/TFLite from flooding the game console.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np

try:  # Keep the console readable; we print our own diagnostics.
    cv2.setLogLevel(0)
except Exception:
    pass

from gesture_code import direction_of, hand_code, is_start_pose  # noqa: E402

# ---------------------------------------------------------------------------
# Tuning constants, identical to the reference two-hand implementation.  They
# are gathered here so they can be tuned on a real camera without touching the
# maths below.
# ---------------------------------------------------------------------------

#: PIP joint angle (degrees) above which the middle phalanx is straight.
FOLD_ANGLE = 158.0
#: DIP joint angle (degrees) above which the distal phalanx is straight.
FOLD_ANGLE_DIP = 140.0
#: Thumb tip distance from the palm centre, relative to the palm size.  The
#: thumb folds by rotating at its base, which no chain angle can see, so its
#: distance from the palm is used as a second signal.  Resting on the fingers
#: ~0.4-0.65, a spread thumb ~0.9-1.1.
THUMB_PALM_RATIO = 0.72
#: Thumb IP angle (degrees) below which the thumb may be curled.
THUMB_FOLD_ANGLE = 150.0
#: How far the thumb's distal phalanx has to bend *towards the palm* before a
#: small IP angle counts as folding instead of hyperextension.  The signed
#: bend is the projection (in [-1, 1]) of the bend direction on the
#: "thumb base -> palm centre" direction; hyperextension points the other way.
THUMB_FLEXION_MIN = 0.10
#: Horizontal thumb offset (relative to palm size) that is unambiguous.
THUMB_SIDE_RATIO = 0.22
#: Number of frames used by the per-character majority vote.
SMOOTH_WINDOW = 4

#: The recognised hand is used with the right-hand convention of the two-hand
#: game (P = thumb points left, N = thumb points right).
HAND_SIDE = "R"

#: Recognition area: the top-right quarter of the camera frame, scaled to 95 %.
ROI_SCALE = 0.95

#: Colours used by the OpenCV preview (BGR).
SKELETON_COLOR = (0, 226, 255)
THUMB_COLOR = (120, 255, 120)

WRIST = 0
THUMB_MCP, THUMB_IP, THUMB_TIP = 2, 3, 4
INDEX_MCP = 5
MIDDLE_MCP = 9
PALM_POINTS = (0, 5, 9, 13, 17)
FINGER_CHAINS = (
    (5, 6, 7, 8),      # index  (MCP, PIP, DIP, TIP)
    (9, 10, 11, 12),   # middle
    (13, 14, 15, 16),  # ring
    (17, 18, 19, 20),  # pinky
)


# ---------------------------------------------------------------------------
# Geometry helpers (ported 1:1 from gesture_twohands.py)
# ---------------------------------------------------------------------------
def _as_xyz(landmarks):
    """Landmarks (objects with x/y/z) -> float32 array of shape (21, 3)."""
    return np.array([[float(lm.x), float(lm.y),
                      float(getattr(lm, "z", 0.0))] for lm in landmarks],
                    dtype=np.float32)


def _joint_angle(a, b, c):
    """Angle ABC in degrees, measured in 3D."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_value = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0))))


def finger_angles(pts, mcp, pip, dip, tip):
    """Return the PIP and DIP joint angles of one long finger in degrees."""
    return (_joint_angle(pts[mcp], pts[pip], pts[dip]),
            _joint_angle(pts[pip], pts[dip], pts[tip]))


def _finger_extended(pts, mcp, pip, dip, tip):
    """A long finger is extended when both of its joints are almost straight."""
    pip_angle, dip_angle = finger_angles(pts, mcp, pip, dip, tip)
    return pip_angle >= FOLD_ANGLE and dip_angle >= FOLD_ANGLE_DIP


def thumb_bend_towards_palm(pts, palm_center):
    """Signed bend of the thumb's distal phalanx, relative to the palm.

    Positive means the tip curls towards the palm (flexion, i.e. folding),
    negative means it bends away from the palm (hyperextension, still out).
    """
    proximal = pts[THUMB_IP] - pts[THUMB_MCP]
    distal = pts[THUMB_TIP] - pts[THUMB_IP]
    n1 = float(np.linalg.norm(proximal))
    n2 = float(np.linalg.norm(distal))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    unit_proximal = proximal / n1
    unit_distal = distal / n2
    bend = unit_distal - unit_proximal * float(
        np.dot(unit_distal, unit_proximal))
    towards_palm = palm_center - pts[THUMB_MCP]
    n3 = float(np.linalg.norm(towards_palm))
    if n3 < 1e-9:
        return 0.0
    return float(np.dot(bend, towards_palm / n3))


def _thumb_extended(pts, palm_center, palm_size):
    """The thumb is out when it is straight and clearly away from the palm."""
    angle = _joint_angle(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP])
    flexed = (angle < THUMB_FOLD_ANGLE and
              thumb_bend_towards_palm(pts, palm_center) > THUMB_FLEXION_MIN)
    if flexed:
        return False
    distance = float(np.linalg.norm(pts[THUMB_TIP] - palm_center)) / palm_size
    return distance >= THUMB_PALM_RATIO


def _thumb_points_right(pts, palm_size):
    """True when the thumb points to the right of the image (mirrored view)."""
    palm_center = pts[list(PALM_POINTS)].mean(axis=0)
    dx = float(pts[THUMB_TIP][0] - palm_center[0])
    if abs(dx) >= THUMB_SIDE_RATIO * palm_size:
        return dx > 0.0
    # Thumb pointing up / down (e.g. a fist): fall back to the side of the
    # palm the thumb is attached to.
    return float(pts[THUMB_MCP][0] - pts[INDEX_MCP][0]) > 0.0


def classify_hand(landmarks):
    """Classify one MediaPipe hand into ``(thumb_token, bits)``.

    The hand is interpreted with the right-hand convention of the two-hand
    game: ``P`` = thumb points left, ``N`` = thumb points right.

    Returns:
        ``(thumb, bits)`` such as ``("P", "11111")`` or ``None`` when the
        landmarks are unusable.
    """
    pts = _as_xyz(landmarks)
    if pts.shape != (21, 3):
        return None
    palm_size = float(np.linalg.norm(pts[WRIST] - pts[MIDDLE_MCP]))
    if palm_size < 1e-6:
        return None
    palm_center = pts[list(PALM_POINTS)].mean(axis=0)

    bits = ["1" if _thumb_extended(pts, palm_center, palm_size) else "0"]
    for mcp, pip, dip, tip in FINGER_CHAINS:
        bits.append("1" if _finger_extended(pts, mcp, pip, dip, tip)
                    else "0")

    pointing_right = _thumb_points_right(pts, palm_size)
    thumb = "P" if not pointing_right else "N"
    return thumb, "".join(bits)


def read_hand(landmarks):
    """Classify one hand and return its 6 character code (or ``None``)."""
    classified = classify_hand(landmarks)
    if classified is None:
        return None
    return hand_code(*classified)


# ---------------------------------------------------------------------------
# Recognition area: top-right quarter of the frame, using 95 % of it
# ---------------------------------------------------------------------------
def roi_rect(frame_shape):
    """Return the recognition ROI as ``(x1, y1, x2, y2)`` pixels.

    The ROI is the top-right quarter (``width/2..width``, ``0..height/2``) of
    the frame, then shrunk to :data:`ROI_SCALE` (95 %) of that quarter: 2.5 %
    of the quarter is cut off on every side.  For a 600x450 preview that is
    exactly ``(308, 6, 592, 219)``.
    """
    height, width = frame_shape[:2]
    quadrant_x = width // 2
    quadrant_w = width - quadrant_x
    quadrant_h = height // 2
    margin_x = int(round(quadrant_w * (1.0 - ROI_SCALE) / 2.0))
    margin_y = int(round(quadrant_h * (1.0 - ROI_SCALE) / 2.0))
    x1 = quadrant_x + margin_x
    y1 = margin_y
    x2 = quadrant_x + quadrant_w - margin_x
    y2 = quadrant_h - margin_y
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Smoothed recognition result
# ---------------------------------------------------------------------------
def _smooth_tokens(history, stable, token):
    """Per-character majority vote over the recent 6 character tokens."""
    history.append(token)
    previous = stable or token
    chars = []
    for index in range(6):
        counts = collections.Counter(item[index] for item in history)
        char, count = counts.most_common(1)[0]
        if count * 2 <= len(history):
            # No strict majority (including a tie): keep what we had.
            char = previous[index]
        chars.append(char)
    return "".join(chars)


class GestureState(object):
    """One recognition result: the smoothed hand code plus derived fields."""

    __slots__ = ("code", "direction", "start_pose", "image", "roi",
                 "timestamp")

    def __init__(self, code=None, image=None, roi=None, timestamp=0.0):
        self.code = code
        self.direction = direction_of(code)
        self.start_pose = is_start_pose(code)
        self.image = image      # annotated camera frame (BGR) or None
        self.roi = roi          # (x1, y1, x2, y2) used for recognition
        self.timestamp = timestamp

    @property
    def hand_count(self):
        return 1 if self.code else 0

    def __repr__(self):
        return "GestureState(code={!r}, direction={!r})".format(
            self.code, self.direction)


#: State used while no camera frame has arrived yet.
EMPTY_STATE = GestureState()


class SingleHandRecognizer(object):
    """MediaPipe Hands wrapper for one hand inside the top-right ROI."""

    def __init__(self,
                 min_detection_confidence=0.55,
                 min_tracking_confidence=0.50,
                 model_complexity=None,
                 smooth_window=SMOOTH_WINDOW):
        import mediapipe as mp

        if model_complexity is None:
            # Complexity 0 is roughly twice as fast and precise enough for one
            # hand; set GAME_HAND_MODEL=1 for the slower, more precise model.
            try:
                model_complexity = int(os.environ.get("GAME_HAND_MODEL", "0"))
            except ValueError:
                model_complexity = 0

        self._modules = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._hands = self._modules.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._history = collections.deque(maxlen=smooth_window)
        self._stable = None

    # -- recognition ----------------------------------------------------
    def _reset(self):
        self._history.clear()
        self._stable = None

    def _smooth(self, token):
        smoothed = _smooth_tokens(self._history, self._stable, token)
        self._stable = smoothed
        return smoothed

    def process(self, frame_bgr):
        """Recognise the hand on *frame_bgr* and draw the skeleton on it.

        The ROI is a view into *frame_bgr*, so the skeleton ends up on the
        frame that is shown in the preview window (the ROI rectangle itself is
        drawn by ``plane_game.annotate_camera``).

        Returns:
            A :class:`GestureState`; ``code`` is ``None`` when no usable hand
            was found.
        """
        if frame_bgr is None or not getattr(frame_bgr, "size", 0):
            return GestureState()

        x1, y1, x2, y2 = roi_rect(frame_bgr.shape)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return GestureState(image=frame_bgr, roi=(x1, y1, x2, y2))

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        code = None
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            classified = classify_hand(hand.landmark)
            if classified is not None:
                code = self._smooth(hand_code(*classified))
                self._draw_hand(roi, hand)
            else:
                self._reset()
        else:
            self._reset()

        return GestureState(code=code, image=frame_bgr, roi=(x1, y1, x2, y2),
                            timestamp=time.time())

    # -- drawing --------------------------------------------------------
    def _draw_hand(self, roi, hand):
        try:
            self._drawing.draw_landmarks(
                roi, hand, self._modules.HAND_CONNECTIONS,
                self._drawing.DrawingSpec(color=SKELETON_COLOR, thickness=2,
                                          circle_radius=3),
                self._drawing.DrawingSpec(color=THUMB_COLOR, thickness=2))
        except Exception:
            # Drawing must never break gesture control.
            return
        points = hand.landmark
        height, width = roi.shape[:2]
        x = int(points[WRIST].x * width)
        y = int(points[WRIST].y * height)
        label = format_label(self._stable)
        label_x = min(max(4, x - 30), max(4, width - 120))
        label_y = min(max(16, y + 18), max(16, height - 26))
        _text_box(roi, label, (label_x, label_y), SKELETON_COLOR, 0.55)

    def close(self):
        try:
            self._hands.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def format_label(code):
    """Short ASCII label for the preview ("P11111 UP")."""
    if not code:
        return "no hand"
    return "{} {}".format(code, direction_of(code).upper())


def _text_box(frame, text, org, color, scale):
    """Draw text with a translucent dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - 4),
                  (x + text_w + 4, y + text_h + baseline + 2),
                  (25, 25, 25), -1)
    cv2.rectangle(frame, (x - 4, y - 4),
                  (x + text_w + 4, y + text_h + baseline + 2),
                  (90, 90, 90), 1)
    cv2.putText(frame, text, (x, y + text_h), font, scale, color,
                thickness, cv2.LINE_AA)


class CameraWorker(object):
    """Capture + recognise in a background thread, expose the newest frame.

    Identical in spirit to the two-hand game: the pygame loop only reads the
    newest annotated frame and :class:`GestureState`, so MediaPipe inference
    never blocks the frame rate.
    """

    def __init__(self, camera, recognizer=None, mirror=True, target_fps=30):
        self._camera = camera
        self._recognizer = recognizer
        self._mirror = mirror
        self._interval = 1.0 / float(target_fps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest_frame = None
        self._latest_gesture = EMPTY_STATE
        self.error = None

    # -- lifecycle ------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run,
                                        name="camera-worker", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- data -----------------------------------------------------------
    def latest(self):
        """Return ``(annotated_image, GestureState)`` (both may be empty)."""
        with self._lock:
            return self._latest_frame, self._latest_gesture

    def _publish(self, image, gesture):
        with self._lock:
            self._latest_frame = image
            self._latest_gesture = gesture

    def _run(self):
        while not self._stop.is_set():
            started = time.time()
            try:
                grabbed, frame = self._camera.read()
            except Exception:
                self.error = "camera read failed"
                traceback.print_exc()
                break
            if not grabbed or frame is None or not getattr(frame, "size", 0):
                time.sleep(0.05)
                continue
            if self._mirror:
                frame = cv2.flip(frame, 1)
            try:
                if self._recognizer is None:
                    gesture = GestureState(image=frame, timestamp=started)
                else:
                    gesture = self._recognizer.process(frame)
            except Exception:
                self.error = "hand recognition failed"
                traceback.print_exc()
                break
            self._publish(gesture.image if gesture.image is not None
                          else frame, gesture)
            delay = self._interval - (time.time() - started)
            if delay > 0:
                time.sleep(delay)

    def close(self):
        self.stop()
        try:
            self._camera.release()
        except Exception:
            pass
        if self._recognizer is not None:
            self._recognizer.close()
