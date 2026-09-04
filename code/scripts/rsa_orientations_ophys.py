"""RSA over drifting_gratings orientations, for every ophys (calcium) subject.

Same per-orientation cosine-similarity RSA as `rsa_orientations.py`, but
built from dF/F fluorescence traces instead of ephys spike times: each
trial's response is the per-ROI mean dF/F within the trial window (rather
than a spike-count rate), averaged across trials for a given orientation.

Runs over every session under `visual_coding_ophys`, skipping sessions that
don't include a drifting_gratings stimulus (e.g. locally_sparse_noise-only
sessions), and saves each session's distance-matrix array and plot named
after the session.
"""

from pathlib import Path

import numpy as np
import pynwb
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

DATA_DIR = Path("/data/visual_coding_ophys")
OUTPUT_DIR = Path("/results/rsa/ophys")

WINDOW = (0.00, 2.00)
STIMULUS_TYPE = "drifting_gratings"

distance_metric = cosine_similarity


def dff_response(
    event_times: np.ndarray,
    timestamps: np.ndarray,
    dff: np.ndarray,
) -> np.ndarray:
    """Per-ROI mean dF/F over the trial window, averaged across trials."""
    start_idx = np.searchsorted(timestamps, event_times + WINDOW[0])
    stop_idx = np.searchsorted(timestamps, event_times + WINDOW[1])
    responses = np.stack(
        [
            dff[start:stop].mean(axis=0)
            for start, stop in zip(start_idx, stop_idx, strict=True)
        ],
    )
    return responses.mean(axis=0)  # (n_cells, )


def compute_session_rsa(session_dir: Path) -> tuple[np.ndarray, list, str] | None:
    nwb_path = next(session_dir.glob("*.nwb.zarr"))
    nwb = pynwb.read_nwb(nwb_path)
    if STIMULUS_TYPE not in nwb.stimulus:
        print(f"{session_dir.name}: no {STIMULUS_TYPE} stimulus, skipping")
        return None

    rrs = (
        nwb.processing["ophys"]
        .data_interfaces["DfOverF"]
        .roi_response_series["DfOverF"]
    )
    timestamps = rrs.timestamps[:]
    dff = rrs.data[:]  # (n_timepoints, n_cells)
    n_cells = dff.shape[1]
    region = next(iter(nwb.imaging_planes.values())).location

    trials = nwb.stimulus[STIMULUS_TYPE].to_dataframe()
    trials = trials[~trials["is_blank_sweep"]]
    orientation_labels = sorted(trials["orientation_in_degrees"].unique())
    n_orientations = len(orientation_labels)

    X_brain = np.zeros((n_orientations, n_cells))
    for i, orientation_label in tqdm(
        enumerate(orientation_labels),
        desc=f"{session_dir.name}: building activations",
        total=n_orientations,
    ):
        event_times = trials.loc[
            trials["orientation_in_degrees"] == orientation_label,
            "start_time",
        ].to_numpy()
        X_brain[i] = dff_response(event_times, timestamps, dff)

    D_brain = np.zeros((n_orientations, n_orientations))
    for i in tqdm(
        range(n_orientations),
        desc=f"{session_dir.name}: computing distances",
    ):
        for j in range(n_orientations):
            D_brain[i, j] = distance_metric(
                torch.from_numpy(X_brain[i]),
                torch.from_numpy(X_brain[j]),
                dim=0,
            ).numpy()

    return D_brain, orientation_labels, region


def main() -> None:
    session_dirs = [p for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
    if not session_dirs:
        raise SystemExit(f"no sessions found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for session_dir in session_dirs:
        result = compute_session_rsa(session_dir)
        if result is None:
            continue
        D_brain, orientation_labels, region = result
        print(D_brain)

        npy_path = OUTPUT_DIR / f"{session_dir.name}.npy"
        np.save(npy_path, D_brain)
        print(f"saved {npy_path}")


if __name__ == "__main__":
    main()
