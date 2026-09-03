"""Standalone drifting-gratings orientation decoder for ophys sessions.

Loads dF/F traces and stimulus tables directly from an NWB file (no
`visual_coding` library import), bins each ROI's mean dF/F into the
drifting-gratings trial window, and cross-validates a linear-SVM decoder.
Runs over every session found in `DATA_DIR` and saves, per session, a result
CSV and an NPZ with the binned activity that fed the decoder.
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

LABEL = "orientation_in_degrees"
WINDOW = (0.00, 2.00)
BIN_SIZE = 0.1
STIMULUS_TYPE = "drifting_gratings"

N_FOLDS = 5
SEED = 0

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "visual_coding_ophys"
OUTPUT_DIR = Path("results/decoding/ophys")


def decode_session(
    session_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]] | None:
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
    trials = trials[~trials["is_blank_sweep"]].dropna(subset=[LABEL])
    event_times = trials["start_time"].to_numpy()
    y = trials[LABEL].to_numpy()

    bin_edges = np.arange(WINDOW[0], WINDOW[1] + BIN_SIZE, BIN_SIZE)
    n_bins = len(bin_edges) - 1

    # (n_trials, n_cells, n_bins) mean dF/F, from per-trial-bin sample averages.
    responses = np.full((len(event_times), n_cells, n_bins), np.nan)
    for t, event in enumerate(event_times):
        start_idx = np.searchsorted(timestamps, event + bin_edges[:-1])
        stop_idx = np.searchsorted(timestamps, event + bin_edges[1:])
        for b, (start, stop) in enumerate(zip(start_idx, stop_idx, strict=True)):
            if stop > start:
                responses[t, :, b] = dff[start:stop].mean(axis=0)

    n_trials = len(event_times)
    X = responses.reshape(n_trials, n_cells * n_bins)
    valid = ~np.isnan(X).any(axis=1)
    responses, X, y, event_times = (
        responses[valid],
        X[valid],
        y[valid],
        event_times[valid],
    )
    print(
        f"{session_dir.name}: {n_cells} {region} cells x {n_bins} bins, "
        f"{len(y)}/{n_trials} trials (dropped {n_trials - len(y)} with empty bins)",
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
                "region": region,
                "label": LABEL,
                "n_cells": n_cells,
                "n_trials": len(y),
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
        "region": region,
        "label": LABEL,
        "responses": responses,  # (n_trials, n_cells, n_bins) mean dF/F
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
