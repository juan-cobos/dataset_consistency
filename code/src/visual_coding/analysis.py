from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visual_coding.dataset import Ephys

RESULTS = Path("/root/capsule/results")
try:
    is_capsule = RESULTS.exists()
except OSError:
    is_capsule = False
if not is_capsule:
    RESULTS = Path(__file__).resolve().parents[3] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def psth(
    neuron: np.ndarray,
    timestamps: np.ndarray | None,
    event_times: np.ndarray,
    window: tuple[float, float] = (-0.5, 1.0),
    bin_size: float = 0.05,
    save_path: Path = RESULTS / "psth.png",
    ylabel: str = "firing rate (Hz)",
) -> pd.Series:
    """Peri-stimulus time histogram around event_times."""
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    centers = bins[:-1] + bin_size / 2

    if timestamps is None:
        counts = np.zeros(len(centers))
        for event in event_times:
            counts += np.histogram(neuron - event, bins=bins)[0]
        response = counts / len(event_times) / bin_size
    else:
        aligned = np.stack(
            [
                np.interp(
                    centers + event,
                    timestamps,
                    neuron,
                    left=np.nan,
                    right=np.nan,
                )
                for event in event_times
            ],
        )
        response = np.nanmean(aligned, axis=0)

    result = pd.Series(response, index=centers, name="psth")

    fig, ax = plt.subplots()
    ax.plot(result.index, result.to_numpy())
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("time from event (s)")
    ax.set_ylabel(ylabel)
    fig.savefig(save_path)
    plt.close(fig)

    return result


if __name__ == "__main__":
    ephys = Ephys()
    session_id = ephys.session_ids()[0]

    trials = ephys.load_trials(session_id)
    flash_times = trials.loc[trials.stimulus_type == "flashes", "start_time"].to_numpy()

    units = ephys.load_units(session_id)
    spike_times = units["spike_times"]

    bin_size = 0.05
    start = min(st.min() for st in spike_times)
    stop = max(st.max() for st in spike_times)
    edges = np.arange(start, stop + bin_size, bin_size)
    centers = edges[:-1] + bin_size / 2

    counts = np.zeros(len(centers))
    for st in spike_times:
        counts += np.histogram(st, bins=edges)[0]
    mean_firing_rate = counts / len(spike_times) / bin_size

    print(psth(mean_firing_rate, centers, flash_times, bin_size=bin_size))
