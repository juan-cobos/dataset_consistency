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


def tuning_curve(
    conditions: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average per-trial responses by condition (e.g. grating orientation), with SEM."""
    conditions = np.asarray(conditions)
    responses = np.asarray(responses, dtype=float)
    unique_conditions = np.unique(conditions)
    grouped = [responses[conditions == c] for c in unique_conditions]
    means = np.array([g.mean() for g in grouped])
    sems = np.array([g.std(ddof=1) / np.sqrt(len(g)) for g in grouped])
    return unique_conditions, means, sems


def selectivity_index(
    conditions: np.ndarray,
    responses: np.ndarray,
    harmonic: int = 2,
) -> float:
    """Vector-sum selectivity of responses across a circular condition (e.g. orientation).

    `harmonic=2` folds the 180-degree symmetry of grating orientation (the
    standard orientation selectivity index); `harmonic=1` treats conditions
    as full 360-degree directions (direction selectivity index). Ranges from
    0 (response spread uniformly across conditions) to 1 (response
    concentrated at a single condition).
    """
    responses = np.clip(np.asarray(responses, dtype=float), 0, None)
    angles = np.deg2rad(np.asarray(conditions, dtype=float)) * harmonic
    vector = np.sum(responses * np.exp(1j * angles))
    total = responses.sum()
    return np.abs(vector) / total if total > 0 else 0.0
