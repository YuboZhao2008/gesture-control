"""Exercise the desktop runtime without touching a native window, camera, or mouse."""

from collections.abc import Callable

import cv2
import numpy as np
import pytest

from gesture_control import cursor_runner, cursor_tracking, runner, vision, windows_mouse
from gesture_control.core.config import AppConfig
from gesture_control.core.types import FrameContext, Gesture, HandFrame, HandLandmarks
from gesture_control.ui.drawing import Image
from gesture_control.vision import CameraFrame

from .test_pipeline import source_hand


def test_cli_defaults_to_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    received: dict[str, object] = {}

    def fake_run(config: AppConfig, **kwargs: object) -> int:
        received.update(kwargs)
        return 3

    monkeypatch.setattr(cursor_runner, "run_cursor", fake_run)
    assert runner.main(["--headless", "--max-frames", "3", "--camera-width", "800"]) == 0
    assert received["headless"] and received["max_frames"] == 3
    assert received["camera_width"] == 800


@pytest.mark.parametrize(
    "stop", ["limit", "escape", "close", "read_error", "preview_error", "headless"]
)
def test_live_cursor_loop(monkeypatch: pytest.MonkeyPatch, stop: str) -> None:
    calls: list[str] = []
    hand = source_hand(Gesture.OPEN_PALM)
    hands = [(hand,), (), (hand,), (hand,)]

    class Camera:
        count = 0

        def __init__(self, **kwargs: object) -> None:
            pass

        def open(self) -> None:
            calls.append("open")

        def read(self) -> CameraFrame:
            if stop == "read_error" and self.count == 1:
                raise RuntimeError("capture failed")
            context = FrameContext(self.count, self.count / 30, 8, 6)
            self.count += 1
            return CameraFrame(np.zeros((6, 8, 3), dtype=np.uint8), context)

        def close(self) -> None:
            calls.append("camera closed")

    class Detector:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def detect(self, image: object, context: FrameContext) -> HandFrame:
            return HandFrame(context, hands[context.frame_id])

        def close(self) -> None:
            calls.append("detector closed")

    class Tracker:
        def update(
            self, observed: HandLandmarks | None, timestamp_s: float, *, image_aspect: float
        ) -> tuple[float, float, bool] | None:
            assert image_aspect == 0.75
            index = round(timestamp_s * 30)
            assert hands[index] == (() if observed is None else (observed,))
            calls.append("track")
            return (0.2, 0.7, True) if observed else None

    class Mouse:
        def __init__(self) -> None:
            assert stop != "headless"

        def update(self, x: float, y: float, pressed: bool) -> None:
            assert (x, y, pressed) == (0.2, 0.7, True)
            calls.append("hold")

        def release(self) -> None:
            calls.append("release")

        def emergency_stop_requested(self) -> bool:
            return stop == "escape" and "hold" in calls

    def capture_preview(*args: object) -> bool:
        calls.append("preview")
        if stop == "preview_error":
            raise RuntimeError("preview failed")
        return stop != "close"

    def record_window(name: str) -> Callable[..., None]:
        def event(*args: object) -> None:
            assert stop != "headless"
            calls.append(name)

        return event

    monkeypatch.setattr(vision, "MediaPipeHandDetector", Detector)
    monkeypatch.setattr(vision, "OpenCVCamera", Camera)
    monkeypatch.setattr(cursor_tracking, "CursorTracker", Tracker)
    monkeypatch.setattr(windows_mouse, "WindowsMouse", Mouse)
    monkeypatch.setattr(cursor_runner, "_preview", capture_preview)
    monkeypatch.setattr(cv2, "getWindowProperty", lambda name, prop: 1)
    monkeypatch.setattr(cv2, "namedWindow", record_window("window"))
    monkeypatch.setattr(cv2, "destroyWindow", record_window("window closed"))

    if stop.endswith("error"):
        with pytest.raises(RuntimeError, match="failed"):
            cursor_runner.run_cursor(AppConfig(), max_frames=3)
    else:
        result = cursor_runner.run_cursor(AppConfig(), max_frames=3, headless=stop == "headless")
        assert result == {"limit": 3, "escape": 2, "close": 1, "headless": 3}[stop]

    if stop == "headless":
        assert calls == ["open", "track", "track", "track", "camera closed", "detector closed"]
    else:
        assert calls[-4:] == ["release", "window closed", "camera closed", "detector closed"]
        assert calls.count("hold") == (2 if stop == "limit" else 1)
        if stop == "limit":
            first = calls.index("hold")
            assert calls[first : first + 7] == [
                "hold",
                "preview",
                "track",
                "release",
                "preview",
                "track",
                "hold",
            ]


def test_preview_tracks_raw_mirrored_wrist(monkeypatch: pytest.MonkeyPatch) -> None:
    hand = source_hand(Gesture.OPEN_PALM, x=0.2, y=0.7)
    context = FrameContext(0, 0, 200, 100)
    captured = CameraFrame(np.zeros((100, 200, 3), dtype=np.uint8), context)
    captured.image[96, 4] = (34, 56, 78)
    rendered: list[Image] = []
    monkeypatch.setattr(cv2, "imshow", lambda name, image: rendered.append(image))
    monkeypatch.setattr(cv2, "pollKey", lambda: 27, raising=False)
    assert not cursor_runner._preview(
        captured, HandFrame(context, (hand,)), (0.97, 0.3, False), False
    )
    image = rendered[0]
    np.testing.assert_array_equal(image[96, 195], [34, 56, 78])
    assert image[69, 159 + 9, 1] > 100
    assert not np.any(image[69, 40])
