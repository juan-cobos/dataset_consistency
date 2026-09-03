"""Drifting-gratings orientation decoding for excitatory vs. inhibitory ephys units.

Loads spikes and stimulus tables directly from an NWB file (no
`visual_coding` library import), splits the good VISp units into
broad-spiking putative excitatory and narrow-spiking putative inhibitory
populations by waveform duration, bins each unit's spike counts into the
drifting-gratings trial window, and cross-validates one linear-SVM decoder
per population. Sessions are only used when both populations clear
MIN_UNITS. Per session it saves a result CSV (one row per population) and an
NPZ with the binned activity that fed each decoder; a final pass plots the
excitatory-vs-inhibitory comparison across sessions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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

# Trough-to-peak duration (ms) splitting narrow-spiking putative inhibitory
# interneurons from broad-spiking putative excitatory cells, as in
# `rsa_orientations_ephys_inhibitory.py`. Sessions are only used when both
# populations have more than MIN_UNITS units.
FS_THRESHOLD = 0.4
MIN_UNITS = 10
POPULATIONS = ["excitatory", "inhibitory"]

# Decoding accuracy grows with population size, and excitatory units far
# outnumber inhibitory ones. With MATCH_UNITS the larger population is
# randomly subsampled to the size of the smaller one so the two decoders see
# equally many units; set to False to decode every unit of each population.
MATCH_UNITS = True

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "visual_coding_neuropixels"
OUTPUT_DIR = Path("results/decoding/ephys_inhib_exc")

COLORS = {"excitatory": "tab:blue", "inhibitory": "tab:red"}


def split_populations(units: pd.DataFrame) -> dict[str, pd.DataFrame] | None:
    """Split units by waveform duration, or None if either population is too small."""
    units = units[units["waveform_duration"].notna()]
    narrow = units["waveform_duration"] < FS_THRESHOLD
    populations = {"inhibitory": units[narrow], "excitatory": units[~narrow]}

    too_few = {
        name: len(group)
        for name, group in populations.items()
        if len(group) <= MIN_UNITS
    }
    if too_few:
        print(f"  {too_few} units, need >{MIN_UNITS} in both populations, skipping")
        return None

    if MATCH_UNITS:
        n_matched = min(len(group) for group in populations.values())
        rng = np.random.default_rng(SEED)
        populations = {
            name: group.iloc[np.sort(rng.choice(len(group), n_matched, replace=False))]
            for name, group in populations.items()
        }
    return populations


def bin_spikes(units: pd.DataFrame, event_times: np.ndarray) -> np.ndarray:
    """(n_trials, n_units, n_bins) firing rate (Hz), from per-trial spike counts."""
    bin_edges = np.arange(WINDOW[0], WINDOW[1] + BIN_SIZE, BIN_SIZE)
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
    return responses.transpose(1, 0, 2)


def decode(responses: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cross-validated linear-SVM accuracy on flattened (unit x bin) features."""
    n_trials = responses.shape[0]
    X = responses.reshape(n_trials, -1)
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pipeline = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    return cross_val_score(pipeline, X, y, cv=cv)


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

    print(f"{session_dir.name}:")
    populations = split_populations(units)
    if populations is None:
        return None

    stimulus_key = f"{STIMULUS_TYPE}_presentations"
    if stimulus_key not in nwb.intervals:
        print(f"{session_dir.name}: no {stimulus_key} interval, skipping")
        return None

    trials = nwb.intervals[stimulus_key].to_dataframe()
    trials = trials.dropna(subset=[LABEL])
    event_times = trials["start_time"].to_numpy()
    y = trials[LABEL].to_numpy()
    chance = 1 / len(np.unique(y))

    rows = []
    activity = {
        "session_id": session_dir.name,
        "region": REGION,
        "label": LABEL,
        "labels": y,
        "event_times": event_times,
        "bin_edges": np.arange(WINDOW[0], WINDOW[1] + BIN_SIZE, BIN_SIZE),
    }
    for population, group in populations.items():
        responses = bin_spikes(group, event_times)
        n_trials, n_units, n_bins = responses.shape
        scores = decode(responses, y)
        print(
            f"  {population}: {n_units} units x {n_bins} bins, {n_trials} trials, "
            f"accuracy {scores.mean():.3f} +/- {scores.std():.3f} "
            f"({scores.mean() / chance:.2f}x chance)",
        )
        rows.append(
            {
                "session_id": session_dir.name,
                "region": REGION,
                "label": LABEL,
                "population": population,
                "n_units": n_units,
                "n_trials": n_trials,
                "accuracy": scores.mean(),
                "std": scores.std(),
                "chance": chance,
                "x_chance": scores.mean() / chance,
                **{f"fold_{i}": score for i, score in enumerate(scores)},
            },
        )
        # (n_trials, n_units, n_bins) firing rate (Hz)
        activity[f"responses_{population}"] = responses

    return pd.DataFrame(rows), activity


def plot_comparison(results: pd.DataFrame) -> None:
    """Per-session paired accuracies and their distributions, exc vs. inhib."""
    accuracy = results.pivot(
        index="session_id",
        columns="population",
        values="accuracy",
    ).dropna()
    if accuracy.empty:
        print("no sessions with both populations decoded, not plotting")
        return

    exc = accuracy["excitatory"].to_numpy()
    inhib = accuracy["inhibitory"].to_numpy()
    chance = results["chance"].iloc[0]
    difference = exc - inhib
    print(
        f"\n{len(accuracy)} sessions: excitatory {exc.mean():.3f}, "
        f"inhibitory {inhib.mean():.3f}, "
        f"mean difference {difference.mean():+.3f} "
        f"(excitatory higher in {np.sum(difference > 0)}/{len(difference)})",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: paired per-session accuracies
    for i, (name, values) in enumerate(zip(POPULATIONS, [exc, inhib])):
        axes[0].scatter(
            np.full(len(values), i)
            + np.random.default_rng(SEED).normal(0, 0.02, len(values)),
            values,
            color=COLORS[name],
            zorder=3,
            label=f"{name} (mean = {values.mean():.3f})",
        )
    axes[0].plot(
        np.tile([[0], [1]], len(exc)),
        np.stack([exc, inhib]),
        color="gray",
        linewidth=0.8,
        alpha=0.6,
        zorder=2,
    )
    axes[0].axhline(chance, color="k", linestyle=":", linewidth=1, label="chance")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(POPULATIONS)
    axes[0].set_ylabel("Decoding accuracy")
    axes[0].set_title(f"{LABEL} decoding in {REGION}, paired by session")
    axes[0].legend(fontsize=8)

    # Right: excitatory vs. inhibitory accuracy against the unity line
    axes[1].scatter(inhib, exc, color="tab:purple", zorder=3)
    lims = [
        min(chance, inhib.min(), exc.min()) - 0.02,
        max(inhib.max(), exc.max()) + 0.02,
    ]
    axes[1].plot(lims, lims, color="k", linestyle=":", linewidth=1)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel("Inhibitory accuracy")
    axes[1].set_ylabel("Excitatory accuracy")
    axes[1].set_title("Excitatory vs. inhibitory accuracy")
    axes[1].text(
        0.02,
        0.95,
        f"mean difference = {difference.mean():+.3f}\n"
        f"excitatory higher in {np.sum(difference > 0)}/{len(difference)}",
        transform=axes[1].transAxes,
        va="top",
    )

    fig.tight_layout()
    png_path = OUTPUT_DIR / "excitatory_vs_inhibitory.png"
    fig.savefig(png_path, dpi=150)
    print(f"saved {png_path}")


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

    csv_paths = sorted(OUTPUT_DIR.glob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"no decoding results found in {OUTPUT_DIR}")
    plot_comparison(pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True))
    plt.show()


if __name__ == "__main__":
    main()
