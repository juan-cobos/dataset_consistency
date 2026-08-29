from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from visual_coding.analysis import zscore
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
    neuro = zscore(neuro)

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


def polar_orientation_tuning(
    orientations: np.ndarray,
    responses: np.ndarray,
    errors: np.ndarray,
    selectivity: float,
    save_path: Path = RESULTS / "polar_orientation_tuning.png",
    ylabel: str = "response",
) -> None:
    """Polar tuning curve of mean response (+/- SEM) across orientation/direction."""
    theta = np.deg2rad(orientations)
    theta = np.append(theta, theta[0] + 2 * np.pi)
    response = np.append(responses, responses[0])
    error = np.append(errors, errors[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(theta, response, marker="o")
    ax.fill_between(theta, response - error, response + error, alpha=0.3)
    ax.set_title(f"selectivity index = {selectivity:.2f}")
    ax.set_ylabel(ylabel, labelpad=30)
    fig.savefig(save_path)
    plt.close(fig)


def orientation_tuning(
    orientations: np.ndarray,
    responses: np.ndarray,
    errors: np.ndarray,
    selectivity: float,
    save_path: Path = RESULTS / "orientation_tuning.png",
    ylabel: str = "response",
) -> None:
    """Linear tuning curve of mean response (+/- SEM) across orientation/direction."""
    fig, ax = plt.subplots()
    ax.errorbar(orientations, responses, yerr=errors, marker="o")
    ax.set_title(f"selectivity index = {selectivity:.2f}")
    ax.set_xlabel("orientation (deg)")
    ax.set_xticks(orientations)
    ax.set_ylabel(ylabel)
    fig.savefig(save_path)
    plt.close(fig)
