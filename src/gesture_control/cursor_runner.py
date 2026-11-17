"""Camera loop for wrist movement and pinch-controlled desktop dragging."""

from __future__ import annotations

import logging
from contextlib import ExitStack, nullcontext
from pathlib import Path

import cv2

from .core.config import AppConfig
from .core.performance import PerformanceMonitor
from .core.types import HandFrame, LandmarkIndex
from .runner import _cleanup, _validate_stream, _vision_resources
from .vision import CameraFrame, draw_hand_frame

LOGGER = logging.getLogger(__name__)
WINDOW = "Gesture Control - Desktop Cursor"


def _window_visible() -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False  # HighGUI may already have discarded the native window.


def _close_preview() -> None:
    if _window_visible():
        cv2.destroyWindow(WINDOW)


def _preview(
    captured: CameraFrame,
    frame: HandFrame,
    pointer: tuple[float, float, bool] | None,
    diagnostics: bool,
) -> bool:
    image = captured.image
    if diagnostics:
        draw_hand_frame(image, frame)
    preview = cv2.flip(image, 1)
    status = "Show your hand"
    if pointer is not None and frame.hands:
        wrist = frame.hands[0].landmarks[LandmarkIndex.WRIST]
        width, height = frame.context.image_width, frame.context.image_height
        center = (
            round((1 - min(max(wrist.x, 0), 1)) * (width - 1)),
            round(min(max(wrist.y, 0), 1) * (height - 1)),
        )
        color = (60, 170, 255) if pointer[2] else (90, 230, 130)
        cv2.circle(preview, center, 9, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.circle(preview, center, 9, color, 2, cv2.LINE_AA)
        status = "Left button held" if pointer[2] else "Moving cursor"
    for y, label in (
        (24, status),
        (48, "Pinch to click or drag | Separate to release | Esc to quit"),
    ):
        cv2.putText(preview, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(preview, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imshow(WINDOW, preview)
    poll_key = getattr(cv2, "pollKey", None)
    key = (poll_key() if callable(poll_key) else cv2.waitKey(1)) & 0xFF
    return key not in (27, ord("q")) and _window_visible()


def run_cursor(
    config: AppConfig,
    *,
    model_path: str | Path = "models/hand_landmarker.task",
    max_frames: int | None = None,
    headless: bool = False,
    performance: PerformanceMonitor | None = None,
    camera_width: int = 640,
    camera_height: int = 480,
    latest_frame: bool = True,
    diagnostics: bool | None = None,
    sensitivity: float = 3.0,
) -> int:
    """Track wrists and control the mouse. Headless runs skip all desktop effects."""
    _validate_stream(max_frames, headless, camera_width, camera_height)
    from .cursor_tracking import CursorTracker

    tracker = CursorTracker(config.smoothing, config.recognition, sensitivity=sensitivity)
    diagnostics = config.debug if diagnostics is None else diagnostics
    with ExitStack() as resources:
        detector, camera = _vision_resources(
            resources, config, model_path, camera_width, camera_height, latest_frame
        )
        indicator = None
        mouse = None
        if not headless:
            from .hold_indicator import HoldIndicator
            from .windows_mouse import WindowsMouse

            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            resources.callback(_cleanup, "cursor preview", _close_preview)
            indicator = HoldIndicator()
            resources.callback(indicator.close)
            mouse = WindowsMouse()
            # Register last to release before waiting for camera shutdown, even on errors.
            resources.callback(mouse.release)
        camera.open()
        count = 0
        while max_frames is None or count < max_frames:
            with performance.frame() if performance is not None else nullcontext():
                with performance.stage("capture") if performance is not None else nullcontext():
                    captured = camera.read()
                with performance.stage("detection") if performance is not None else nullcontext():
                    frame = detector.detect(captured.image, captured.context)
                hand = frame.hands[0] if frame.hands else None
                pointer = tracker.update(
                    hand,
                    frame.context.timestamp_s,
                    image_aspect=frame.context.image_height / frame.context.image_width,
                )
                count += 1
                if mouse is not None:
                    if mouse.emergency_stop_requested() or (count > 1 and not _window_visible()):
                        break
                    if pointer is None:
                        mouse.release()
                    else:
                        mouse.update(*pointer)
                    assert indicator is not None
                    indicator.show(pointer is not None and pointer[2])
                    if not _preview(captured, frame, pointer, diagnostics):
                        break
        LOGGER.info("Processed %d cursor frames", count)
        return count
