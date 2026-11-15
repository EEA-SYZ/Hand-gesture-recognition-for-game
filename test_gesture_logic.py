# -*- coding: utf-8 -*-
"""Unit tests for the finger-distance gesture classifier (no camera needed)."""

from types import SimpleNamespace

from gesture_mediapipe import classify_landmarks


def make_hand(open_flags, thumb="closed"):
    """Build 21 synthetic landmarks in a palm-up coordinate system."""
    pts = [(0.5, 0.92)] * 21
    pts[0] = (0.50, 0.92)
    # thumb: 1 CMC, 2 MCP, 3 IP, 4 tip
    pts[1] = (0.42, 0.78)
    pts[2] = (0.35, 0.70)
    if thumb == "left":
        pts[3], pts[4] = (0.28, 0.66), (0.18, 0.63)
    elif thumb == "right":
        pts[3], pts[4] = (0.72, 0.66), (0.82, 0.63)
    elif thumb == "up":
        pts[3], pts[4] = (0.38, 0.60), (0.40, 0.45)
    else:  # folded across the palm
        pts[3], pts[4] = (0.44, 0.66), (0.48, 0.60)

    bases = ((5, 0.38, 0.62), (9, 0.50, 0.58),
             (13, 0.62, 0.62), (17, 0.73, 0.67))
    for idx, (mcp, x, y) in enumerate(bases):
        pts[mcp] = (x, y)
        if open_flags[idx]:
            pts[mcp + 1] = (x, y - 0.14)
            pts[mcp + 2] = (x, y - 0.26)
            pts[mcp + 3] = (x, y - 0.38)
        else:
            pts[mcp + 1] = (x, y - 0.03)
            pts[mcp + 2] = (x + 0.01, y + 0.02)
            pts[mcp + 3] = (x + 0.01, y + 0.04)
    return [SimpleNamespace(x=x, y=y, z=0.0) for x, y in pts]


CASES = [
    ("open palm", make_hand([True, True, True, True], "left"), "up"),
    ("fist", make_hand([False, False, False, False], "closed"), "down"),
    ("thumb left", make_hand([False, False, False, False], "left"), "left"),
    ("thumb right", make_hand([False, False, False, False], "right"), "right"),
    ("thumb up (other)", make_hand([False, False, False, False], "up"),
     "pause"),
    ("point (other)", make_hand([True, False, False, False], "closed"),
     "pause"),
    ("peace (other)", make_hand([True, True, False, False], "closed"),
     "pause"),
]


def main():
    failed = 0
    for name, hand, expected in CASES:
        got = classify_landmarks(hand)
        ok = got == expected
        print("[{}] {:<18} expected={:<5} got={}".format(
            "PASS" if ok else "FAIL", name, expected, got))
        failed += 0 if ok else 1
    print("RESULT:", "all passed" if not failed else
          "{} failed".format(failed))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
