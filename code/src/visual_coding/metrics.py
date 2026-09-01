import numpy as np

from visual_coding.analysis import trial_spike_rates


def condition_means(
    conditions: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean response magnitude per unique condition."""
    conditions = np.asarray(conditions)
    responses = np.asarray(responses, dtype=float)
    unique_conditions, condition_ids = np.unique(conditions, return_inverse=True)
    means = np.bincount(condition_ids, weights=responses) / np.bincount(condition_ids)
    return unique_conditions, means


def preferred_condition(
    conditions: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-trial responses for the condition eliciting the largest mean response."""
    conditions = np.asarray(conditions)
    responses = np.asarray(responses, dtype=float)
    unique_conditions, means = condition_means(conditions, responses)
    preferred = unique_conditions[int(np.argmax(means))]
    return preferred, responses[conditions == preferred]


def preferred_parameter(
    values: np.ndarray,
    responses: np.ndarray,
) -> float:
    """Stimulus parameter value eliciting the largest mean response, marginalizing over the others."""
    unique_values, means = condition_means(values, responses)
    return unique_values[int(np.argmax(means))]


def lifetime_sparseness(responses: np.ndarray) -> float:
    """Lifetime sparseness (Vinje and Gallant, 2000) over per-condition mean responses.

    1 when only a single condition drives a non-zero response (maximally
    selective), 0 when every condition drives an equal response. Being
    nonparametric it applies to any stimulus type, unlike `selectivity_index`,
    which needs conditions varying along one circular parameter.
    """
    responses = np.clip(np.asarray(responses, dtype=float), 0, None)
    n = len(responses)
    sum_squares = np.sum(responses**2)
    if n < 2 or sum_squares == 0:
        return 0.0
    return float((1 - responses.sum() ** 2 / (n * sum_squares)) / (1 - 1 / n))


def spontaneous_response_distribution(
    spike_times: np.ndarray,
    spontaneous_window: tuple[float, float],
    duration: float,
    n_intervals: int = 1000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mean firing rate (Hz) from the spontaneous epoch."""
    rng = np.random.default_rng() if rng is None else rng
    start, stop = spontaneous_window
    starts = rng.uniform(start, stop - duration, size=n_intervals)
    return trial_spike_rates(spike_times, starts, duration)


def participation_ratio(values: np.ndarray) -> float:
    """Participation ratio normalized."""
    values = np.clip(np.asarray(values, dtype=float), 0, None)
    mean_square = np.mean(values**2)
    if values.size == 0 or mean_square == 0:
        return 0.0
    return float(np.mean(values) ** 2 / mean_square)


def dim_cov(neuro: np.ndarray) -> float:
    """Dimensionality of the covariance."""
    eigenvalues = np.linalg.eigvalsh(np.cov(neuro, rowvar=False))
    return participation_ratio(eigenvalues)


def dim_rates(neuro: np.ndarray) -> float:
    """Dimensionality of the rates."""
    return participation_ratio(neuro.mean(axis=0))
