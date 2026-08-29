import ast

import numpy as np
import pandas as pd
from PIL import Image

from visual_coding.dataset import Ephys
from visual_coding.utils import RESULTS


def _parse(value: object) -> object:
    """Presentation columns like size/color/phase are stored as string reprs."""
    return ast.literal_eval(value) if isinstance(value, str) else value


def _scalar(value: object) -> float:
    parsed = _parse(value)
    return float(parsed[0]) if isinstance(parsed, list | tuple) else float(parsed)


def _size(value: object) -> tuple[int, int]:
    height, width = _parse(value)
    return int(height), int(width)


def _to_rgb(color: object) -> np.ndarray:
    """Map a psychopy-style [-1, 1] color (scalar or RGB triple) to [0, 1] RGB."""
    rgb = np.atleast_1d(np.asarray(_parse(color), dtype=float))
    if rgb.size == 1:
        rgb = np.repeat(rgb, 3)
    return (rgb + 1) / 2


def _sinusoidal_grating(
    size: tuple[int, int],
    orientation: float,
    spatial_frequency: float,
    phase: float,
    contrast: float,
) -> np.ndarray:
    """Full-field sinusoidal grating, luminance in [0, 1] (1 pixel per degree)."""
    height, width = size
    y, x = np.mgrid[0:height, 0:width]
    theta = np.deg2rad(orientation)
    cycles = x * np.cos(theta) + y * np.sin(theta)
    return 0.5 + 0.5 * contrast * np.sin(
        2 * np.pi * (spatial_frequency * cycles + phase),
    )


def static_grating(presentation: pd.Series) -> Image.Image:
    """Render a row of `static_gratings_presentations` to an image."""
    grating = _sinusoidal_grating(
        _size(presentation["size"]),
        presentation["orientation"],
        _scalar(presentation["spatial_frequency"]),
        _scalar(presentation["phase"]),
        presentation["contrast"],
    )
    rgb = grating[..., None] * _to_rgb(presentation["color"])
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))


def drifting_grating(presentation: pd.Series, phase: float = 0.0) -> Image.Image:
    """Render a single frame (at `phase` cycles) of `drifting_gratings_presentations`."""
    grating = _sinusoidal_grating(
        _size(presentation["size"]),
        presentation["orientation"],
        _scalar(presentation["spatial_frequency"]),
        phase,
        presentation["contrast"],
    )
    rgb = grating[..., None] * _to_rgb(presentation["color"])
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))


FIELD_SIZE = (250, 250)


def gabor(
    presentation: pd.Series,
    phase: float = 0.0,
    field_size: tuple[int, int] = FIELD_SIZE,
) -> Image.Image:
    """Render a single frame (at `phase` cycles) of `gabors_presentations`, gray-padded to `field_size`."""
    height, width = _size(presentation["size"])
    grating = _sinusoidal_grating(
        (height, width),
        presentation["orientation"],
        _scalar(presentation["spatial_frequency"]),
        phase,
        presentation["contrast"],
    )
    yy, xx = np.mgrid[0:height, 0:width]
    aperture = (xx - width / 2) ** 2 + (yy - height / 2) ** 2 <= (
        min(height, width) / 2
    ) ** 2

    rgb = grating[..., None] * _to_rgb(presentation["color"])
    rgb = np.where(aperture[..., None], rgb, 0.5)
    patch = (rgb * 255).clip(0, 255).astype(np.uint8)

    field_height, field_width = field_size
    canvas = np.full((field_height, field_width, 3), 0.5 * 255, dtype=np.uint8)
    top, left = (field_height - height) // 2, (field_width - width) // 2
    canvas[top : top + height, left : left + width] = patch
    return Image.fromarray(canvas)


def flash(presentation: pd.Series) -> Image.Image:
    """Render a row of `flashes_presentations`: a solid full-field color."""
    size = _size(presentation["size"])
    rgb = np.ones((*size, 3)) * _to_rgb(presentation["color"])
    return Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8))


if __name__ == "__main__":
    ephys = Ephys()
    session_id = ephys.session_ids()[0]
    trials = ephys.load_trials(session_id)

    static_grating(trials[trials.stimulus_type == "static_gratings"].iloc[0]).save(
        RESULTS / "static_grating.png",
    )
    drifting_grating(trials[trials.stimulus_type == "drifting_gratings"].iloc[0]).save(
        RESULTS / "drifting_grating.png",
    )
    gabor(trials[trials.stimulus_type == "gabors"].iloc[0]).save(RESULTS / "gabor.png")
    flash(trials[trials.stimulus_type == "flashes"].iloc[0]).save(RESULTS / "flash.png")
