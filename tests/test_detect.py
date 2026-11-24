"""Unit tests for the parts of detect.py that don't need a camera or dlib.

Run with:  python -m pytest tests
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detect import FatigueMonitor, eye_aspect_ratio, mouth_aspect_ratio  # noqa: E402


def test_open_eye_has_higher_ear_than_closed_eye():
    open_eye = [(0, 0), (1, -1), (2, -1), (3, 0), (2, 1), (1, 1)]
    closed_eye = [(0, 0), (1, -0.1), (2, -0.1), (3, 0), (2, 0.1), (1, 0.1)]
    assert eye_aspect_ratio(open_eye) > 0.5
    assert eye_aspect_ratio(closed_eye) < 0.1


def test_open_mouth_has_higher_mar():
    closed = [(0, 0), (1, -0.1), (2, -0.1), (3, -0.1), (4, 0), (3, 0.1), (2, 0.1), (1, 0.1)]
    wide = [(0, 0), (1, -2), (2, -2), (3, -2), (4, 0), (3, 2), (2, 2), (1, 2)]
    assert mouth_aspect_ratio(closed) < 0.1
    assert mouth_aspect_ratio(wide) > 0.6


def test_drowsy_alert_fires_once_after_enough_closed_frames():
    m = FatigueMonitor(ear_thresh=0.25, ear_frames=5)
    events = [m.update(0.1, 0.0) for _ in range(10)]
    assert events[:4] == [[]] * 4
    assert events[4] == ["drowsy"]
    assert all(e == [] for e in events[5:])  # no repeat while eyes stay closed
    assert m.drowsy and m.drowsy_events == 1
    m.update(0.3, 0.0)
    assert not m.drowsy


def test_short_closure_counts_as_blink_not_drowsiness():
    m = FatigueMonitor(ear_thresh=0.25, ear_frames=20)
    for _ in range(3):
        m.update(0.1, 0.0)
    m.update(0.3, 0.0)
    assert m.blinks == 1 and m.drowsy_events == 0


def test_yawn_counted_once_per_yawn():
    m = FatigueMonitor(mar_thresh=0.6, yawn_frames=3)
    for _ in range(2):
        for _ in range(6):
            m.update(0.3, 0.9)
        m.update(0.3, 0.1)
    assert m.yawns == 2
