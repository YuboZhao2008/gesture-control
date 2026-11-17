from dataclasses import replace

import pytest

from gesture_control.core.types import FrameContext, Gesture, HandLandmarks, LandmarkIndex
from gesture_control.cursor_tracking import CursorTracker
from gesture_control.training.augmentation import source_observation
from gesture_control.training.fixtures import synthetic_samples

_CONTEXT = FrameContext(1, 0, 640, 480)
_ASPECT = _CONTEXT.image_height / _CONTEXT.image_width


def _hand(gesture: Gesture, *, x: float = 0.5, y: float = 0.575) -> HandLandmarks:
    sample = next(sample for sample in synthetic_samples() if sample.label is gesture)
    return replace(source_observation(sample, _CONTEXT, translation=(x, y)), hand_id="hand")


def _held(tracker: CursorTracker, hand: HandLandmarks, timestamp_s: float) -> bool:
    cursor = tracker.update(hand, timestamp_s, image_aspect=_ASPECT)
    assert cursor is not None
    return cursor[2]


@pytest.mark.parametrize(
    ("x", "y", "screen"),
    [
        (0.5, 0.575, (0.5, 0.5)),
        (0.15, 0.30, (1.0, 0.0)),
        (0.85, 0.85, (0.0, 1.0)),
        (1.2, -0.1, (0.0, 0.0)),
    ],
)
def test_cursor_uses_mirrored_wrist_inside_comfortable_bounds(
    x: float, y: float, screen: tuple[float, float]
) -> None:
    cursor = CursorTracker().update(_hand(Gesture.POINT, x=x, y=y), 0, image_aspect=_ASPECT)
    assert cursor is not None
    assert cursor[:2] == pytest.approx(screen)
    assert cursor[2] is False


def test_fist_debounces_both_transitions_and_ignores_thumb_contact() -> None:
    tracker = CursorTracker()
    fist = _hand(Gesture.FIST)
    points = list(fist.landmarks)
    points[LandmarkIndex.THUMB_TIP] = points[LandmarkIndex.INDEX_TIP]
    fist = replace(fist, landmarks=tuple(points))
    open_hand = _hand(Gesture.OPEN_PALM)

    assert not _held(tracker, fist, 0)
    assert not _held(tracker, open_hand, 0.03)
    assert not _held(tracker, fist, 0.06)
    assert not _held(tracker, fist, 0.10)
    assert _held(tracker, fist, 0.13)
    assert _held(tracker, open_hand, 0.15)
    assert _held(tracker, fist, 0.18)
    assert _held(tracker, open_hand, 0.20)
    assert not _held(tracker, open_hand, 0.25)


def test_hand_loss_releases_and_requires_new_dwell() -> None:
    tracker = CursorTracker()
    fist = _hand(Gesture.FIST)

    assert not _held(tracker, fist, 0)
    assert _held(tracker, fist, 0.07)
    assert tracker.update(None, 0.08) is None
    assert not _held(tracker, fist, 0.10)
    assert _held(tracker, fist, 0.17)


@pytest.mark.parametrize(("hand_id", "arrival"), [("new-hand", 0.08), ("hand", 0.50)])
def test_identity_change_or_long_gap_releases(hand_id: str, arrival: float) -> None:
    tracker = CursorTracker()
    fist = _hand(Gesture.FIST)

    assert not _held(tracker, fist, 0)
    assert _held(tracker, fist, 0.07)
    assert not _held(tracker, replace(fist, hand_id=hand_id), arrival)


def test_collapsed_finger_releases_immediately() -> None:
    tracker = CursorTracker()
    fist = _hand(Gesture.FIST)
    points = list(fist.landmarks)
    points[LandmarkIndex.INDEX_PIP] = points[LandmarkIndex.INDEX_MCP]
    malformed = replace(fist, landmarks=tuple(points))

    assert not _held(tracker, fist, 0)
    assert _held(tracker, fist, 0.07)
    assert not _held(tracker, malformed, 0.08)


def test_pinch_does_not_trigger_click() -> None:
    tracker = CursorTracker()
    pinch = _hand(Gesture.PINCH)

    assert not _held(tracker, pinch, 0)
    assert not _held(tracker, pinch, 0.10)
