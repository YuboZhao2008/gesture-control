from dataclasses import replace

import pytest

from gesture_control.core.config import SmoothingConfig
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


def test_tap_locks_target_and_drag_starts_only_after_deliberate_motion() -> None:
    tracker = CursorTracker(SmoothingConfig(enabled=False))

    def sample(gesture: Gesture, time: float, x: float) -> tuple[float, float, bool]:
        result = tracker.update(_hand(gesture, x=x), time, image_aspect=_ASPECT)
        assert result is not None
        return result

    aim = sample(Gesture.OPEN_PALM, 0, 0.5)
    assert sample(Gesture.PINCH, 0.03, 0.502) == aim
    down = sample(Gesture.PINCH, 0.065, 0.504)
    assert down == (*aim[:2], True)
    assert sample(Gesture.PINCH, 0.09, 0.501) == down
    assert sample(Gesture.OPEN_PALM, 0.12, 0.511) == aim
    sample(Gesture.OPEN_PALM, 0.14, 0.511)
    sample(Gesture.PINCH, 0.16, 0.511)
    assert sample(Gesture.PINCH, 0.20, 0.511)[2]
    dragged = sample(Gesture.PINCH, 0.24, 0.65)
    assert dragged[0] < aim[0] - 0.1 and dragged[2]
    assert sample(Gesture.OPEN_PALM, 0.28, 0.66) == (*dragged[:2], False)


@pytest.mark.parametrize("break_type", ["missing", "new-hand", "time-gap", "collapsed"])
def test_tracking_discontinuity_requires_open_before_repress(break_type: str) -> None:
    tracker = CursorTracker()
    pinch = _hand(Gesture.PINCH)
    opened = _hand(Gesture.OPEN_PALM)
    assert not _held(tracker, pinch, 0)
    assert not _held(tracker, pinch, 0.1)
    assert not _held(tracker, opened, 0.13)
    assert not _held(tracker, pinch, 0.16)
    assert _held(tracker, pinch, 0.20)
    now = 0.24
    if break_type == "missing":
        assert tracker.update(None, 0.23) is None
    elif break_type == "new-hand":
        pinch = replace(pinch, hand_id="other")
        opened = replace(opened, hand_id="other")
    elif break_type == "time-gap":
        now = 0.60
    else:
        points = list(pinch.landmarks)
        points[LandmarkIndex.MIDDLE_MCP] = points[LandmarkIndex.WRIST]
        assert not _held(tracker, replace(pinch, landmarks=tuple(points)), 0.23)
    assert not _held(tracker, pinch, now)
    assert not _held(tracker, pinch, now + 0.05)
    assert not _held(tracker, opened, now + 0.10)
    assert not _held(tracker, pinch, now + 0.15)
    assert _held(tracker, pinch, now + 0.20)


def test_single_noisy_contact_cannot_press() -> None:
    tracker = CursorTracker()
    opened, pinch = _hand(Gesture.OPEN_PALM), _hand(Gesture.PINCH)
    for frame, hand in enumerate((opened, pinch, opened, pinch, opened)):
        assert not _held(tracker, hand, frame / 30)


def test_release_noise_cannot_start_a_second_press() -> None:
    tracker = CursorTracker()
    opened, pinch = _hand(Gesture.OPEN_PALM), _hand(Gesture.PINCH)
    poses = [opened, pinch, pinch, opened, pinch, pinch, opened, opened, pinch, pinch]
    states = [_held(tracker, hand, frame / 30) for frame, hand in enumerate(poses)]
    assert states == [False, False, True, False, False, False, False, False, False, True]


@pytest.mark.parametrize("adaptive", [True, False])
def test_wrist_jitter_and_reacquisition(adaptive: bool) -> None:
    tracker = CursorTracker(SmoothingConfig(adaptive=adaptive))
    raw = CursorTracker(SmoothingConfig(enabled=False))
    positions: list[float] = []
    raw_positions: list[float] = []
    for index in range(60):
        x = 0.503 if index % 2 else 0.497
        hand = _hand(Gesture.OPEN_PALM, x=x)
        smooth = tracker.update(hand, index / 30)
        direct = raw.update(hand, index / 30)
        assert smooth and direct
        if index >= 30:
            positions.append(smooth[0])
            raw_positions.append(direct[0])
    assert max(positions) - min(positions) < 0.0024
    assert max(positions) - min(positions) < (max(raw_positions) - min(raw_positions)) / 3
    tracker.update(None, 2)
    hand = _hand(Gesture.OPEN_PALM, x=0.8)
    reacquired = tracker.update(hand, 2.1)
    assert reacquired is not None and reacquired[0] == positions[-1]


@pytest.mark.parametrize("fps", [20, 30, 60])
def test_fast_sweep_travels_far_and_slow_corrections_stay_precise(fps: int) -> None:
    def travel(distance: float, duration: float) -> float:
        tracker = CursorTracker()
        frames = round(duration * fps)
        for i in range(frames + fps):
            x = 0.5 - distance * min(i / frames, 1)
            cursor = tracker.update(_hand(Gesture.OPEN_PALM, x=x), i / fps)
        assert cursor is not None
        return cursor[0] - 0.5

    assert 0.46 < travel(0.12, 0.3) <= 0.5
    assert 0.005 < travel(0.006, 1) < 0.007


def test_screen_edge_reverses_without_traversing_unused_camera_area() -> None:
    tracker = CursorTracker()
    for i in range(21):
        p = tracker.update(_hand(Gesture.OPEN_PALM, x=0.5 - i * 0.015), i / 30)
    assert p and p[0] == 1
    for i in range(1, 6):
        p = tracker.update(_hand(Gesture.OPEN_PALM, x=0.2 + i * 0.004), (20 + i) / 30)
    assert p and p[0] < 0.99
