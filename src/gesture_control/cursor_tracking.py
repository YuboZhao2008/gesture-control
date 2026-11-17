"""Translate one unmirrored hand observation into desktop cursor intent."""

from math import isfinite

from gesture_control.core.types import HandLandmarks, Landmark3D, LandmarkIndex
from gesture_control.recognition.geometry import finger_extension_score, landmark_distance

_FINGER_BASES = (
    LandmarkIndex.INDEX_MCP,
    LandmarkIndex.MIDDLE_MCP,
    LandmarkIndex.RING_MCP,
    LandmarkIndex.PINKY_MCP,
)
_FIST_ENTER_SCORE = 0.55
_FIST_EXIT_SCORE = 0.65
_FIST_ENTER_S = 0.060
_FIST_EXIT_S = 0.040
_MAX_TRACK_GAP_S = 0.25
_CAMERA_X_BOUNDS = (0.15, 0.85)
_CAMERA_Y_BOUNDS = (0.30, 0.85)


class CursorTracker:
    """Use the wrist for motion and all four fingers curled for the left button."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Forget the prior hand and discard any pending or held fist."""
        self._hand_id: str | None = None
        self._last_seen_s: float | None = None
        self._transition_since_s: float | None = None
        self._fist_held = False

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
        if (
            self._hand_id != hand.hand_id
            or self._last_seen_s is None
            or not 0 <= timestamp_s - self._last_seen_s <= _MAX_TRACK_GAP_S
        ):
            self.reset()
        self._hand_id, self._last_seen_s = hand.hand_id, timestamp_s

        extension = _max_finger_extension(hand, image_aspect)
        if extension is None:
            # Collapsed chains provide no evidence of a fist.
            self._fist_held = False
            self._transition_since_s = None
        else:
            threshold = _FIST_EXIT_SCORE if self._fist_held else _FIST_ENTER_SCORE
            fist = extension <= threshold
            dwell_s = _FIST_ENTER_S if fist else _FIST_EXIT_S
            if fist == self._fist_held:
                self._transition_since_s = None
            elif self._transition_since_s is None:
                self._transition_since_s = timestamp_s
            elif timestamp_s >= self._transition_since_s + dwell_s:
                self._fist_held = fist
                self._transition_since_s = None

        wrist = hand.landmarks[LandmarkIndex.WRIST]
        x = _screen_fraction(1.0 - wrist.x, *_CAMERA_X_BOUNDS)
        y = _screen_fraction(wrist.y, *_CAMERA_Y_BOUNDS)
        return x, y, self._fist_held


def _screen_fraction(value: float, lower: float, upper: float) -> float:
    return min(1.0, max(0.0, (value - lower) / (upper - lower)))


def _max_finger_extension(hand: HandLandmarks, image_aspect: float) -> float | None:
    # Keep y and depth in the same image-width units as x for joint geometry.
    maximum = 0.0
    for base in _FINGER_BASES:
        mcp, pip, dip, tip = (
            Landmark3D(p.x, p.y * image_aspect, p.z) for p in hand.landmarks[base : base + 4]
        )
        lengths = (
            landmark_distance(mcp, pip),
            landmark_distance(pip, dip),
            landmark_distance(dip, tip),
        )
        if min(lengths) <= max(1e-12, 0.05 * max(lengths)):
            return None
        maximum = max(maximum, finger_extension_score(mcp, pip, dip, tip))
    return maximum
