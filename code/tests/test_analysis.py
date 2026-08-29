"""Validate our selectivity_index against the dataset's precomputed g_osi_dg.

The Ephys units table ships a precomputed global orientation selectivity
index (`g_osi_dg`) and each unit's preferred drifting-gratings temporal
frequency (`pref_tf_dg`). This recomputes OSI from raw spike times via
`average_firing_rate`/`align_epochs`/`tuning_curve`/`selectivity_index` for
a sample of units and checks it against the dataset's value.
"""

import numpy as np
import pytest

from visual_coding.analysis import (
    align_epochs,
    average_firing_rate,
    selectivity_index,
    tuning_curve,
)
from visual_coding.dataset import Ephys

WINDOW = (0.0, 2.0)
BIN_SIZE = 0.05
NUM_UNITS = 200


@pytest.fixture(scope="module")
def drifting_gratings() -> tuple:
    ephys = Ephys()
    session_id = ephys.session_ids()[0]

    trials = ephys.load_trials(session_id)
    gratings = trials[trials.stimulus_type == "drifting_gratings"].dropna(
        subset=["orientation"],
    )
    units = ephys.load_units(session_id).dropna(subset=["g_osi_dg", "pref_tf_dg"])
    return gratings, units.iloc[:NUM_UNITS]


def _our_osi(
    spike_times: np.ndarray,
    rate_timestamps: np.ndarray,
    start: float,
    stop: float,
    event_times: np.ndarray,
    orientations: np.ndarray,
) -> float:
    firing_rate = average_firing_rate([spike_times], BIN_SIZE, start, stop)
    epochs, _ = align_epochs(
        firing_rate,
        rate_timestamps,
        event_times,
        WINDOW,
        BIN_SIZE,
    )
    responses = np.nanmean(epochs, axis=1)
    conditions, means, _ = tuning_curve(orientations, responses)
    return selectivity_index(conditions, means)


def test_selectivity_index_matches_dataset_g_osi_dg(drifting_gratings: tuple) -> None:
    gratings, units = drifting_gratings
    event_times = gratings["start_time"].to_numpy()
    orientations = gratings["orientation"].to_numpy()
    temporal_frequencies = gratings["temporal_frequency"].to_numpy()

    start = event_times.min() + WINDOW[0]
    stop = event_times.max() + WINDOW[1]
    rate_edges = np.arange(start, stop + BIN_SIZE, BIN_SIZE)
    rate_timestamps = rate_edges[:-1] + BIN_SIZE / 2

    ours = np.array(
        [
            _our_osi(
                unit["spike_times"],
                rate_timestamps,
                start,
                stop,
                event_times[temporal_frequencies == unit["pref_tf_dg"]],
                orientations[temporal_frequencies == unit["pref_tf_dg"]],
            )
            for _, unit in units.iterrows()
        ],
    )
    dataset = units["g_osi_dg"].to_numpy()

    assert np.corrcoef(ours, dataset)[0, 1] > 0.99
    assert np.abs(ours - dataset).mean() < 0.02
