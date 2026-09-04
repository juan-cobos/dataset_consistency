import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "visual_coding_neuropixels"
FEATURE_DIR = ROOT / "results" / "features"
IMAGE_DIR = FEATURE_DIR / "drifting_gratings_images"

STIMULUS_TYPE = "drifting_gratings"
LABEL = "orientation"

MODELS = ("microsoft/resnet-50",)  # ("facebook/dinov2-base", "microsoft/resnet-50")
BATCH_SIZE = 32

# What the animal saw is fixed by these columns; kept in the trial table so a
# repeated frame can still be recognised downstream.
CONDITION_COLUMNS = [
    "orientation",
    "spatial_frequency",
    "phase",
    "contrast",
    "size",
    "color",
]


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


def onset_phase(presentation: pd.Series) -> float:
    """Phase of the grating at trial onset, in cycles.

    The table records the phase the stimulus software had accumulated, which
    runs to five figures; only its fractional part sets what is on the screen.
    """
    return _scalar(presentation["phase"]) % 1.0


def get_stimuli_per_session(
    session_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray] | None:
    """Presentations of `STIMULUS_TYPE`, and the frame each one showed."""
    nwb_path = next(session_dir.glob("*.nwb.zarr"))
    nwb = pynwb.read_nwb(nwb_path)

    stimulus_key = f"{STIMULUS_TYPE}_presentations"
    if stimulus_key not in nwb.intervals:
        print(f"{session_dir.name}: no {stimulus_key} interval, skipping")
        return None

    trials = nwb.intervals[stimulus_key].to_dataframe()
    trials = trials.dropna(subset=[LABEL])
    trials = trials.sort_values("start_time")

    conditions = trials[CONDITION_COLUMNS].astype(str).agg("|".join, axis=1)
    trials = trials.assign(condition=conditions)

    frames = np.stack(
        [
            np.asarray(drifting_grating(presentation, onset_phase(presentation)))
            for _, presentation in trials.iterrows()
        ],
    )
    return trials, frames


def save_images(session_id: str, frames: np.ndarray, trials: pd.DataFrame) -> Path:
    """Write one PNG per presentation, plus the order they were presented in."""
    session_image_dir = IMAGE_DIR / session_id / "images"
    session_image_dir.mkdir(parents=True, exist_ok=True)

    filenames = [f"image_{i:04d}.png" for i in range(len(frames))]
    for filename, frame in zip(filenames, frames, strict=True):
        Image.fromarray(frame).save(session_image_dir / filename)

    presentations = pd.DataFrame(
        {
            "presentation": np.arange(len(frames)),
            "start_time": trials["start_time"].to_numpy(dtype=float),
            "stop_time": trials["stop_time"].to_numpy(dtype=float),
            "orientation": trials[LABEL].to_numpy(dtype=float),
            "temporal_frequency": trials["temporal_frequency"].to_numpy(dtype=float),
            "condition": trials["condition"].to_numpy(),
            "filename": filenames,
        },
    )
    presentations.to_csv(session_image_dir / "presentations.csv", index=False)
    return session_image_dir


def pool(hidden: torch.Tensor) -> torch.Tensor:
    """One vector per image, averaging over whatever the spatial axes are.

    A transformer block is (batch, tokens, features) and averages over tokens; a
    convnet stage is (batch, channels, height, width) and averages over the
    feature map, leaving the channels as the features.
    """
    if hidden.ndim == 4:
        return hidden.mean(dim=(2, 3))
    return hidden.mean(dim=1)


def extract_features(
    images: list[np.ndarray],
    processor: AutoImageProcessor,
    model: AutoModel,
) -> dict[str, np.ndarray]:
    """Pooled hidden states of each image: {block: (n_images, n_features)}.

    block_0 is the model's own input embedding - patch embeddings for a ViT,
    the stem for a convnet - and the last is its final block.
    """
    parts: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(images), BATCH_SIZE):
        batch = images[start : start + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt").to(model.device)
        with torch.no_grad():
            hidden = model(**inputs, output_hidden_states=True).hidden_states
        for i, block in enumerate(hidden):
            parts.setdefault(f"block_{i}", []).append(
                pool(block).float().cpu().numpy(),
            )
    return {block: np.concatenate(chunks) for block, chunks in parts.items()}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}")

    models = {}
    for model_name in MODELS:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).eval().to(device)
        models[model_name] = (processor, model)
        print(f"loaded {model_name}")

    session_dirs = [p for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
    if not session_dirs:
        raise SystemExit(f"no sessions found in {DATA_DIR}")

    for model_name in MODELS:
        (FEATURE_DIR / model_name.split("/")[-1]).mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for session_dir in session_dirs:
        npz_paths = {
            model_name: FEATURE_DIR
            / model_name.split("/")[-1]
            / f"{session_dir.name}.npz"
            for model_name in MODELS
        }
        pending = [name for name, path in npz_paths.items() if not path.exists()]
        if not pending:
            print(f"{session_dir.name}: features already exist, skipping")
            continue

        stimuli = get_stimuli_per_session(session_dir)
        if stimuli is None:
            continue
        trials, frames = stimuli

        height, width = frames.shape[1:3]
        print(
            f"{session_dir.name[:24]}: {len(frames)} presentations, "
            f"{trials['condition'].nunique()} distinct conditions, "
            f"{width}x{height} px",
        )

        session_image_dir = save_images(session_dir.name, frames, trials)
        print(f"  saved {len(frames)} PNGs to {session_image_dir}")

        for model_name in pending:
            processor, model = models[model_name]
            features = extract_features(list(frames), processor, model)
            np.savez_compressed(
                npz_paths[model_name],
                model=model_name,
                blocks=np.arange(len(features)),
                conditions=trials["condition"].to_numpy().astype(str),
                orientation=trials[LABEL].to_numpy(dtype=float),
                temporal_frequency=trials["temporal_frequency"].to_numpy(dtype=float),
                start_time=trials["start_time"].to_numpy(dtype=float),
                stop_time=trials["stop_time"].to_numpy(dtype=float),
                **features,
            )
            widths = {name: matrix.shape[1] for name, matrix in features.items()}
            print(
                f"  {model_name}: {len(features)} blocks, "
                f"features {min(widths.values())}-{max(widths.values())}, "
                f"saved {npz_paths[model_name]}",
            )


if __name__ == "__main__":
    main()
