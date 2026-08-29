from collections.abc import Sequence

import numpy as np


def zscore(neuro: np.ndarray) -> np.ndarray:
    """Standardize neuro to zero mean and unit variance."""
    neuro = np.asarray(neuro, dtype=float)
    mean = neuro.mean(axis=0, keepdims=True)
    std = neuro.std(axis=0, keepdims=True)
    return (neuro - mean) / np.where(std == 0, 1, std)


def average_firing_rate(
    spike_times: Sequence[np.ndarray],
    bin_size: float = 0.05,
    start: float | None = None,
    stop: float | None = None,
) -> np.ndarray:
    """Bin ragged per-unit spike times into a population-averaged firing rate (Hz)."""
    if start is None:
        start = min(st.min() for st in spike_times)
    if stop is None:
        stop = max(st.max() for st in spike_times)
    edges = np.arange(start, stop + bin_size, bin_size)

    counts = np.zeros(len(edges) - 1)
    for st in spike_times:
        counts += np.histogram(st, bins=edges)[0]
    return counts / len(spike_times) / bin_size


def align_epochs(
    neuro: np.ndarray,
    timestamps: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float],
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Align a continuous neuro trace onto a window around each event time."""
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = bins[:-1] + bin_size / 2

    epochs = np.stack(
        [
            np.interp(centers + event, timestamps, neuro, left=np.nan, right=np.nan)
            for event in event_times
        ],
    )
    return epochs, centers
