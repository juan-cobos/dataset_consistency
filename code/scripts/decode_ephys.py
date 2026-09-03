"""Standalone drifting-gratings orientation decoder for ephys sessions.

Loads spikes and stimulus tables directly from an NWB file (no
`visual_coding` library import), bins each unit's spike counts into the
drifting-gratings trial window, and cross-validates a linear-SVM decoder.
Runs over every session found in `DATA_DIR` and saves, per session, a result
CSV and an NPZ with the binned spike-count activity that fed the decoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LABEL = "orientation"
REGION = "VISp"
WINDOW = (0.00, 2.00)
BIN_SIZE = 0.1
STIMULUS_TYPE = "drifting_gratings"

N_FOLDS = 5
SEED = 0

AMPLITUDE_CUTOFF = 0.1
PRESENCE_RATIO = 0.95
ISI_VIOLATIONS = 0.5

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "visual_coding_neuropixels"
OUTPUT_DIR = Path("results/decoding/ephys")


def decode_session(
    session_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]] | None:
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

    stimulus_key = f"{STIMULUS_TYPE}_presentations"
    if stimulus_key not in nwb.intervals:
        print(f"{session_dir.name}: no {stimulus_key} interval, skipping")
        return None

    trials = nwb.intervals[stimulus_key].to_dataframe()
    trials = trials.dropna(subset=[LABEL])
    event_times = trials["start_time"].to_numpy()
    y = trials[LABEL].to_numpy()

    bin_edges = np.arange(WINDOW[0], WINDOW[1] + BIN_SIZE, BIN_SIZE)
    n_bins = len(bin_edges) - 1

    # (n_units, n_trials, n_bins) firing rate (Hz), from per-trial spike counts.
    responses = np.stack(
        [
            np.stack(
                [
                    np.histogram(spike_times, bins=event + bin_edges)[0]
                    for event in event_times
                ],
            )
            / BIN_SIZE
            for spike_times in units["spike_times"]
        ],
    )
    n_units, n_trials, _ = responses.shape
    responses = responses.transpose(1, 0, 2)  # (n_trials, n_units, n_bins)
    X = responses.reshape(n_trials, n_units * n_bins)
    print(
        f"{session_dir.name}: {n_units} {REGION} units x {n_bins} bins, {n_trials} trials",
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    scores = cross_val_score(pipeline, X, y, cv=cv)

    chance = 1 / len(np.unique(y))
    print(
        f"{session_dir.name}: accuracy {scores.mean():.3f} +/- {scores.std():.3f} "
        f"({scores.mean() / chance:.2f}x chance)",
    )

    results = pd.DataFrame(
        [
            {
                "session_id": session_dir.name,
                "region": REGION,
                "label": LABEL,
                "n_units": n_units,
                "n_trials": n_trials,
                "accuracy": scores.mean(),
                "std": scores.std(),
                "chance": chance,
                "x_chance": scores.mean() / chance,
                **{f"fold_{i}": score for i, score in enumerate(scores)},
            },
        ],
    )
    activity = {
        "session_id": session_dir.name,
        "region": REGION,
        "label": LABEL,
        "responses": responses,  # (n_trials, n_units, n_bins) firing rate (Hz)
        "labels": y,
        "event_times": event_times,
        "bin_edges": bin_edges,
    }
    return results, activity


def main() -> None:
    session_dirs = [p for p in sorted(DATA_DIR.iterdir()) if p.is_dir()]
    if not session_dirs:
        raise SystemExit(f"no sessions found in {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for session_dir in session_dirs:
        csv_path = OUTPUT_DIR / f"{session_dir.name}.csv"
        npz_path = OUTPUT_DIR / f"{session_dir.name}.npz"
        if csv_path.exists() and npz_path.exists():
            print(f"{session_dir.name}: {csv_path} already exists, skipping")
            continue

        result = decode_session(session_dir)
        if result is None:
            continue
        results, activity = result
        results.to_csv(csv_path, index=False)
        np.savez_compressed(npz_path, **activity)
        print(f"saved {csv_path}")
        print(f"saved {npz_path}")


if __name__ == "__main__":
    main()
