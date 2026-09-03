"""RSA over drifting_gratings orientations, for every ephys subject/session.

Same population-vector / cosine-similarity RSA as `rsa.py`, but instead of
one activation vector per stimulus type (drifting_gratings, gabors, ...) it
builds one vector per distinct drifting_gratings orientation, averaging
firing rate over that orientation's repeated presentations (pooled across
temporal frequencies).

Runs over every session under `visual_coding_neuropixels`, skipping
sessions with no good units in `REGION`, and saves each session's
distance-matrix array and plot named after the session.
"""

from pathlib import Path

import numpy as np
import pynwb
import torch
from torch.nn.functional import cosine_similarity
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "visual_coding_neuropixels"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "rsa" / "ephys"

REGION = "VISp"
WINDOW = (0.00, 2.00)
STIMULUS_TYPE = "drifting_gratings"

AMPLITUDE_CUTOFF = 0.1
PRESENCE_RATIO = 0.95
ISI_VIOLATIONS = 0.5

distance_metric = cosine_similarity


def firing_rates(
    event_times: np.ndarray,
    spike_trains: np.ndarray,
) -> np.ndarray:
    """Per-unit firing rate from spike counts over the trial window."""
    duration = WINDOW[1] - WINDOW[0]
    starts = event_times + WINDOW[0]
    stops = event_times + WINDOW[1]
    counts = np.stack(
        [
            np.searchsorted(spike_times, stops) - np.searchsorted(spike_times, starts)
            for spike_times in spike_trains
        ],
    )
    return (counts / duration).mean(axis=1)  # (n_cells, )


def compute_session_rsa(session_dir: Path) -> tuple[np.ndarray, list] | None:
    nwb_path = next(session_dir.glob("*.nwb.zarr"))
    nwb = pynwb.read_nwb(nwb_path)

    units = nwb.units.to_dataframe()
    units = units[
        (units["quality"] == "good")
        & (units["amplitude_cutoff"] <= AMPLITUDE_CUTOFF)
        & (units["presence_ratio"] >= PRESENCE_RATIO)
        & (units["isi_violations"] <= ISI_VIOLATIONS)
        & (units["ecephys_structure_acronym"] == REGION)
    ]
    if units.empty:
        print(f"{session_dir.name}: no good units in {REGION}, skipping")
        return None

    presentations_key = f"{STIMULUS_TYPE}_presentations"
    if presentations_key not in nwb.intervals:
        print(f"{session_dir.name}: no {presentations_key} interval table, skipping")
        return None

    trials = (
        nwb.intervals[presentations_key].to_dataframe().dropna(subset=["orientation"])
    )
    if trials.empty:
        print(f"{session_dir.name}: no {STIMULUS_TYPE} stimulus, skipping")
        return None

    orientation_labels = sorted(trials["orientation"].unique())
    n_orientations = len(orientation_labels)
    n_cells = len(units)

    X_brain = np.zeros((n_orientations, n_cells))
    for i, orientation_label in tqdm(
        enumerate(orientation_labels),
        desc=f"{session_dir.name}: building activations",
        total=n_orientations,
    ):
        event_times = trials.loc[
            trials["orientation"] == orientation_label,
            "start_time",
        ].to_numpy()
        X_brain[i] = firing_rates(event_times, units["spike_times"])

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

    return D_brain, orientation_labels


def main() -> None:
    session_dirs = [p for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
    if not session_dirs:
        raise SystemExit(f"no sessions found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for session_dir in session_dirs:
        npy_path = OUTPUT_DIR / f"{session_dir.name}.npy"
        if npy_path.exists():
            print(f"{session_dir.name}: {npy_path} already exists, skipping")
            continue

        result = compute_session_rsa(session_dir)
        if result is None:
            continue
        D_brain, orientation_labels = result
        print(D_brain)

        np.save(npy_path, D_brain)
        print(f"saved {npy_path}")


if __name__ == "__main__":
    main()
