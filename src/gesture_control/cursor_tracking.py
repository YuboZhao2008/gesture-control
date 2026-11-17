"""Translate one unmirrored hand observation into desktop cursor intent."""

from math import dist, isfinite

from gesture_control.core.config import RecognitionConfig, SmoothingConfig
from gesture_control.core.types import HandLandmarks, LandmarkIndex
from gesture_control.temporal.filtering import OneEuroFilter, PointerFilterConfig

_PINCH_CONFIRM_S = 0.030
_DRAG_DISTANCE = 0.012  # fraction of the usable screen span
_MAX_TRACK_GAP_S = 0.25
_CAMERA_X_BOUNDS = (0.15, 0.85)
_CAMERA_Y_BOUNDS = (0.30, 0.85)


class CursorTracker:
    """Use the wrist for motion and a thumb-index pinch for the left button."""

    def __init__(
        self, smoothing: SmoothingConfig | None = None, recognition: RecognitionConfig | None = None
    ) -> None:
        self.recognition = recognition or RecognitionConfig()
        self.smoothing = smoothing or SmoothingConfig()
        self._filter = OneEuroFilter(
            PointerFilterConfig(
                min_cutoff=self.smoothing.min_cutoff,
                beta=self.smoothing.beta,
                derivative_cutoff=self.smoothing.derivative_cutoff,
            )
        )
        self.reset()

    def reset(self) -> None:
        """Forget the prior hand and discard any pending or held pinch."""
        self._filter.reset()
        self._wrist: tuple[float, float] | None = None
        self._hand_id: str | None = None
        self._last_seen_s: float | None = None
        self._transition_since_s: float | None = None
        self._armed = False
        self._pressed = False
        self._dragging = False
        self._anchor = (0.0, 0.0)
        self._start_position = (0.0, 0.0)
        self._output: tuple[float, float] | None = None

    def update(
        self, hand: HandLandmarks | None, timestamp_s: float, *, image_aspect: float = 1.0
    ) -> tuple[float, float, bool] | None:
        """Return desktop cursor fractions and the debounced held state.

        The mirrored camera region is inset so the wrist can reach screen edges
        while leaving room in the camera frame for the fingers.

        Set image_aspect to the source height divided by width. A missing hand
        returns None and immediately clears the held state. A new identity or
        discontinuous timestamp starts the debounce again.
        """
        if not isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and nonnegative")
        if not isfinite(image_aspect) or image_aspect <= 0:
            raise ValueError("image_aspect must be finite and positive")
        if hand is None:
            self.reset()
            return None
        release_at = None
        if (
            self._hand_id != hand.hand_id
            or self._last_seen_s is None
            or not 0 <= timestamp_s - self._last_seen_s <= _MAX_TRACK_GAP_S
        ):
            if self._pressed:
                release_at = self._output
            self.reset()
        self._hand_id = hand.hand_id

        wrist = hand.landmarks[LandmarkIndex.WRIST]
        wx, wy = wrist.x, wrist.y
        if self.smoothing.enabled:
            if self.smoothing.adaptive:
                wx, wy = self._filter.filter(wx, wy, timestamp_s)
            elif self._wrist is not None and self._last_seen_s is not None:
                alpha = 1 - (1 - self.smoothing.alpha) ** (30 * (timestamp_s - self._last_seen_s))
                wx = self._wrist[0] + alpha * (wx - self._wrist[0])
                wy = self._wrist[1] + alpha * (wy - self._wrist[1])
        self._wrist = wx, wy
        self._last_seen_s = timestamp_s
        x = _screen_fraction(1.0 - wx, *_CAMERA_X_BOUNDS)
        y = _screen_fraction(wy, *_CAMERA_Y_BOUNDS)
        position = (x, y)
        distance = _pinch_ratio(hand, image_aspect)
        if distance is None or distance >= self.recognition.pinch_exit_threshold:
            # Deliver button-up at the last emitted location even if release moves the wrist.
            if self._pressed and self._output is not None:
                position = self._output
            # After releasing, require another separated sample before rearming.
            # A single noisy open frame cannot generate a second click.
            self._armed = distance is not None and not self._pressed
            self._pressed = self._dragging = False
            self._transition_since_s = None
        elif self._armed:
            if self._pressed:
                if dist(position, self._start_position) >= _DRAG_DISTANCE:
                    self._dragging = True
                if not self._dragging:
                    position = self._anchor
            elif distance <= self.recognition.pinch_enter_threshold:
                if self._transition_since_s is None:
                    self._transition_since_s = timestamp_s
                    self._anchor = self._output or position
                    self._start_position = position
                elif timestamp_s - self._transition_since_s >= _PINCH_CONFIRM_S:
                    self._pressed = True
                position = self._anchor
            else:
                self._transition_since_s = None
        if release_at is not None:
            position = release_at
        self._output = position
        return *position, self._pressed


def _screen_fraction(value: float, lower: float, upper: float) -> float:
    return min(1.0, max(0.0, (value - lower) / (upper - lower)))


def _pinch_ratio(hand: HandLandmarks, aspect: float) -> float | None:
    def coordinates(index: LandmarkIndex) -> tuple[float, float, float]:
        point = hand.landmarks[index]
        return point.x, point.y * aspect, point.z

    palm = dist(coordinates(LandmarkIndex.WRIST), coordinates(LandmarkIndex.MIDDLE_MCP))
    if palm < 1e-6:
        return None
    return dist(coordinates(LandmarkIndex.THUMB_TIP), coordinates(LandmarkIndex.INDEX_TIP)) / palm
