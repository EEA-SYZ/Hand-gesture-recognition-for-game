# -*- coding: utf-8 -*-
"""Unit tests for the single-hand gesture logic (no camera needed).

The synthetic 21 landmarks are built exactly like the ones in the reference
two-hand project (``Hand-gesture-ps-game/test_rhythm_logic.py``), so the
ported classifier is checked against the same kind of input.

Run with::

    .venv\\Scripts\\python.exe test_gesture_logic.py
"""

from __future__ import absolute_import

from types import SimpleNamespace

from gesture_code import (direction_of, hand_code, is_start_pose, parse_hand,
                          format_code)
from gesture_single import (CONFIRM_FRAMES, GestureConfirmer, classify_hand,
                            read_hand, roi_rect)

PALM_WRIST = (0.50, 0.90)
PALM_MIDDLE = (0.48, 0.59)

FAILED = []


def check(name, ok, detail=""):
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             " - " + detail if detail else ""))
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------------------
# Synthetic hands
# ---------------------------------------------------------------------------
def build_hand(open_flags, thumb_right=True, thumb_out=True):
    """Synthetic 21 landmarks in image coordinates (y grows downwards).

    Args:
        open_flags: four booleans for index, middle, ring, pinky.
        thumb_right: True when the thumb points to the right of the image.
        thumb_out: True when the thumb is extended, False when it is folded.
    """
    points = [(0.5, 0.8)] * 21
    points[0] = PALM_WRIST
    xs = [0.55, 0.48, 0.41, 0.34]  # index..pinky when the thumb is right
    if not thumb_right:
        xs = [1 - value for value in xs][::-1]
    for (mcp, _name), x, is_open in zip(
            ((5, "index"), (9, "middle"), (13, "ring"), (17, "pinky")),
            xs, open_flags):
        points[mcp] = (x, 0.60)
        if is_open:
            points[mcp + 1] = (x, 0.47)
            points[mcp + 2] = (x, 0.40)
            points[mcp + 3] = (x, 0.34)
        else:
            points[mcp + 1] = (x, 0.57)
            points[mcp + 2] = (x, 0.60)
            points[mcp + 3] = (x, 0.63)
    sign = 1.0 if thumb_right else -1.0
    base_x = 0.5 + sign * 0.10
    points[1] = (base_x, 0.74)
    points[2] = (base_x + sign * 0.03, 0.72)
    if thumb_out:
        points[3] = (base_x + sign * 0.07, 0.70)
        points[4] = (base_x + sign * 0.13, 0.68)
    else:
        points[3] = (base_x + sign * 0.01, 0.70)
        points[4] = (base_x - sign * 0.01, 0.69)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in points]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_hand_code():
    check("hand_code joins", hand_code("P", "11111") == "P11111")
    check("hand_code upper-cases the thumb", hand_code("n", "00000") == "N00000")
    check("parse_hand splits", parse_hand("N10000") == ("N", "10000"))
    for bad in ("P1111", "X11111", "P1111a", "P1111", ""):
        try:
            parse_hand(bad)
            ok = False
        except ValueError:
            ok = True
        check("parse_hand rejects {!r}".format(bad), ok)
    check("format_code keeps valid codes", format_code("P10000") == "P10000")
    check("format_code hides invalid codes", format_code("zzz") == "--")


def test_classifier():
    """One hand only, right-hand convention: P = thumb left, N = thumb right."""
    cases = [
        # (name, hand, expected thumb, expected bits)
        ("open hand, thumb left", build_hand((1, 1, 1, 1), False, True),
         "P", "11111"),
        ("open hand, thumb right", build_hand((1, 1, 1, 1), True, True),
         "N", "11111"),
        ("open hand, thumb folded right",
         build_hand((1, 1, 1, 1), True, False), "N", "01111"),
        ("open hand, thumb folded left",
         build_hand((1, 1, 1, 1), False, False), "P", "01111"),
        ("fist, thumb right", build_hand((0, 0, 0, 0), True, False),
         "N", "00000"),
        ("fist, thumb left", build_hand((0, 0, 0, 0), False, False),
         "P", "00000"),
        ("only thumb left", build_hand((0, 0, 0, 0), False, True),
         "P", "10000"),
        ("only thumb right", build_hand((0, 0, 0, 0), True, True),
         "N", "10000"),
        ("point, thumb folded", build_hand((1, 0, 0, 0), True, False),
         "N", "01000"),
        ("peace", build_hand((1, 1, 0, 0), False, False), "P", "01100"),
        ("three fingers", build_hand((1, 1, 1, 0), True, True),
         "N", "11110"),
    ]
    for name, hand, thumb, bits in cases:
        got = classify_hand(hand)
        check("classify " + name, got == (thumb, bits), str(got))
        check("code " + name, read_hand(hand) == thumb + bits)

    check("classify rejects short input",
          classify_hand(build_hand((1, 1, 1, 1))[:10]) is None)


def test_directions():
    """The task mapping: Xx1111 up, X00000 down, P/N10000 left/right."""
    up_codes = ("P11111", "N11111", "P01111", "N01111")
    for code in up_codes:
        check("direction {} is up".format(code), direction_of(code) == "up")

    for code in ("P00000", "N00000"):
        check("direction {} is down".format(code),
              direction_of(code) == "down")

    check("P10000 is left", direction_of("P10000") == "left")
    check("N10000 is right", direction_of("N10000") == "right")

    # Everything else leaves the plane where it is.
    for code in ("P01000", "P00100", "P00001", "P11000", "P01100",
                 "P11110", "N01000", None, "", "P1", "X11111"):
        check("direction {} is pause".format(code),
              direction_of(code) == "pause")


def test_start_pose():
    check("P11111 starts", is_start_pose("P11111"))
    check("N11111 starts", is_start_pose("N11111"))
    for code in ("P01111", "N01111",    # thumb folded
                 "P00000", "N00000",    # fist
                 "P10000", "N10000",    # thumb only
                 "P11110", "P11101",    # one long finger folded
                 "P11111P11111",        # two-hand code is not a hand code
                 None, ""):
        check("{!r} does not start".format(code), not is_start_pose(code))


def test_pose_to_direction():
    """Recognised landmarks -> hand code -> direction / start pose."""
    open_left = read_hand(build_hand((1, 1, 1, 1), False, True))
    check("open hand, thumb left is P11111", open_left == "P11111")
    check("P11111 moves up and starts a round",
          direction_of(open_left) == "up" and is_start_pose(open_left))

    folded = read_hand(build_hand((1, 1, 1, 1), True, False))
    check("open hand, thumb folded is N01111", folded == "N01111")
    check("N01111 moves up but does not start a round",
          direction_of(folded) == "up" and not is_start_pose(folded))

    fist = read_hand(build_hand((0, 0, 0, 0), True, False))
    check("fist is down", direction_of(fist) == "down", fist)

    thumb = read_hand(build_hand((0, 0, 0, 0), False, True))
    check("only the left thumb is left", direction_of(thumb) == "left", thumb)

    thumb = read_hand(build_hand((0, 0, 0, 0), True, True))
    check("only the right thumb is right", direction_of(thumb) == "right", thumb)

    point = read_hand(build_hand((1, 0, 0, 0), False, False))
    check("a point does not move the plane", direction_of(point) == "pause",
          point)


def test_confirmation():
    """A reading is only accepted after three identical frames in a row."""
    check("the confirmation window is three frames", CONFIRM_FRAMES == 3)
    confirmer = GestureConfirmer()
    check("one frame is not enough", confirmer.push("P10000") is None)
    check("two frames are not enough", confirmer.push("P10000") is None)
    check("three identical frames confirm the gesture",
          confirmer.push("P10000") == "P10000")

    # A different reading has to start counting from zero again.
    check("a single different frame keeps the confirmed gesture",
          confirmer.push("N10000") == "P10000")
    check("two different frames keep it too",
          confirmer.push("N10000") == "P10000")
    check("its third frame switches the gesture",
          confirmer.push("N10000") == "N10000")

    # Losing the hand is confirmed the same way, so one dropped frame does not
    # stop the plane.
    check("one no-hand frame keeps the gesture",
          confirmer.push(None) == "N10000")
    check("two no-hand frames keep it too",
          confirmer.push(None) == "N10000")
    check("three no-hand frames clear the gesture",
          confirmer.push(None) is None)

    # Flickering between two readings never confirms anything.
    flapping = GestureConfirmer()
    readings = [flapping.push(code)
                for code in ("P10000", "N10000") * 3]
    check("alternating readings never confirm",
          all(reading is None for reading in readings))

    # The window can be configured (tests / tuning).
    fast = GestureConfirmer(frames=2)
    check("the confirmation window is configurable",
          fast.push("P11111") is None and fast.push("P11111") == "P11111")


def test_roi():
    """The ROI is 95 % of the top-right quarter of the frame."""
    width, height = 600, 450
    x1, y1, x2, y2 = roi_rect((height, width, 3))
    check("ROI x starts inside the right half", x1 > width // 2, str(x1))
    check("ROI stays in the top half", y2 < height // 2, str(y2))
    roi_w, roi_h = x2 - x1, y2 - y1
    quarter_w, quarter_h = width - width // 2, height // 2
    check("ROI width is 95% of the quarter",
          abs(roi_w / float(quarter_w) - 0.95) < 0.02,
          "{:.3f}".format(roi_w / float(quarter_w)))
    check("ROI height is 95% of the quarter",
          abs(roi_h / float(quarter_h) - 0.95) < 0.02,
          "{:.3f}".format(roi_h / float(quarter_h)))
    check("ROI margins are symmetric horizontally",
          abs((x1 - width // 2) - (width - x2)) <= 1, str((x1, x2)))
    check("ROI margins are symmetric vertically",
          abs(y1 - (height // 2 - y2)) <= 1, str((y1, y2)))
    check("600x450 ROI is (308, 6, 592, 219)",
          roi_rect((450, 600, 3)) == (308, 6, 592, 219),
          str(roi_rect((450, 600, 3))))

    # A 4:3 / 16:9 frame keeps the ROI inside the frame.
    for shape in ((720, 1280, 3), (1080, 1920, 3), (240, 320, 3)):
        rx1, ry1, rx2, ry2 = roi_rect(shape)
        check("ROI inside {}x{}".format(shape[1], shape[0]),
              0 <= rx1 < rx2 <= shape[1] and 0 <= ry1 < ry2 <= shape[0],
              str((rx1, ry1, rx2, ry2)))


def main():
    test_hand_code()
    test_classifier()
    test_directions()
    test_start_pose()
    test_pose_to_direction()
    test_confirmation()
    test_roi()
    print()
    if FAILED:
        print("RESULT: {} failed: {}".format(len(FAILED), ", ".join(FAILED)))
        return 1
    print("RESULT: all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
