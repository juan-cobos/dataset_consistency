import numpy as np


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


if __name__ == "__main__":
    from visual_coding.dataset import Ephys
    from visual_coding.transforms import AverageFiringRate
    from visual_coding.utils import RESULTS
    from visual_coding.viz import psth

    ephys = Ephys()
    session_id = ephys.session_ids()[0]

    window = (-0.5, 2.0)
    bin_size = 0.05

    trials = ephys.load_trials(session_id)
    flashes = trials.loc[trials.stimulus_type == "flashes"]
    event_times = flashes["start_time"].to_numpy()

    units = ephys.load_units(session_id)

    # Only the flash events (+ window) matter for the PSTH, so bin just that
    # span instead of each region's full-session spike range.
    start = event_times.min() + window[0]
    stop = event_times.max() + window[1]
    rate_edges = np.arange(start, stop + bin_size, bin_size)
    rate_timestamps = rate_edges[:-1] + bin_size / 2

    for region in ephys.brain_structures(session_id):
        spike_times = units.loc[
            units["ecephys_structure_acronym"] == region,
            "spike_times",
        ]
        mean_firing_rate = AverageFiringRate(bin_size)(spike_times, start, stop)
        epochs, centers = align_epochs(
            mean_firing_rate,
            rate_timestamps,
            event_times,
            window,
            bin_size,
        )

        psth(
            epochs,
            centers,
            save_path=RESULTS / f"psth_{region}.png",
            ylabel=f"{region} firing rate (Hz)",
        )
