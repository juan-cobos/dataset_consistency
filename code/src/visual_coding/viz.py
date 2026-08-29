from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual_coding.dataset import Ephys, Ophys
from visual_coding.transforms import ZScore
from visual_coding.utils import RESULTS


def spike_raster(
    spike_times: Sequence[np.ndarray],
    window: tuple[float, float] | None = None,
    save_path: Path = RESULTS / "spike_raster.pdf",
) -> None:
    """Raster plot of per-unit spike times."""
    if window is not None:
        spike_times = [st[(st >= window[0]) & (st <= window[1])] for st in spike_times]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.eventplot(spike_times, linelengths=0.8, color="k")
    if window is not None:
        ax.set_xlim(window)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit")
    fig.savefig(save_path)
    plt.close(fig)


def calcium_traces(
    neuro: np.ndarray,
    timestamps: np.ndarray,
    offset: float = 5.0,
    save_path: Path = RESULTS / "calcium_traces.pdf",
) -> None:
    """Z-scored calcium (dF/F) traces, one per ROI, vertically offset."""
    neuro = ZScore()(neuro)

    fig, ax = plt.subplots(figsize=(10, 6))
    for roi in range(neuro.shape[1]):
        ax.plot(timestamps, neuro[:, roi] + roi * offset, linewidth=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ROI (offset, z-scored)")
    fig.savefig(save_path)
    plt.close(fig)


def psth(
    epochs: np.ndarray,
    centers: np.ndarray,
    save_path: Path = RESULTS / "psth.png",
    ylabel: str = "neuro",
) -> None:
    """Plot mean and SEM across pre-aligned epochs."""
    response = np.nanmean(epochs, axis=0)
    n_trials = np.sum(~np.isnan(epochs), axis=0)
    sem = np.nanstd(epochs, axis=0, ddof=1) / np.sqrt(n_trials)

    fig, ax = plt.subplots()
    ax.plot(centers, response)
    ax.fill_between(centers, response - sem, response + sem, alpha=0.3)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("time from event (s)")
    ax.set_ylabel(ylabel)
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == "__main__":
    ephys = Ephys()
    session_id = ephys.session_ids()[0]
    spike_times = ephys.load_units(session_id)["spike_times"].iloc[:20].tolist()
    start = min(st.min() for st in spike_times)
    spike_raster(spike_times, window=(start, start + 10))

    ophys = Ophys()
    session_id = ophys.session_ids()[0]
    dff = ophys.load_dff(session_id).iloc[:1000, :20]
    calcium_traces(dff.to_numpy(), dff.index.to_numpy())
