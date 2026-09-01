"""One interface over the datasets, so downstream code stops branching on them.

Each dataset records the same experiment differently: spikes or dF/F, one
imaging plane or eight, stimulus parameters in `nwb.intervals` or in
`nwb.stimulus`, running speed as a frame or as a dict of them. `Adapter`
translates all of that into four calls with fixed shapes and column names:

    population(session_id)            (n_time, n_neurons) signal + timestamps
    presentations(session_id, stim)   one row per presentation, canonical names
    behavior(session_id, name)        one behavioral measure, as recorded
    images(session_id, stimulus_type) pictures + a label per presentation

Analyses written against those four work on any dataset, and a dataset is
added by teaching the adapter, not by touching every analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

from visual_coding.analysis import average_firing_rate, zscore
from visual_coding.dataset import Dataset, Ephys, Ophys, VisualLearning
from visual_coding.utils import resolve_capsule_path

IMAGES_PATH = resolve_capsule_path("images")

SHARED_REGION = "VISp"
SHARED_BEHAVIOR = "running_speed"

# Visual learning names each image by identity; visual coding names the set.
IMAGE_NAME = re.compile(r"im\d+", re.IGNORECASE)


def stimulus_family(stimulus_type: str) -> str | None:
    """Group a dataset's own stimulus name into the family the datasets share."""
    name = str(stimulus_type).lower()
    if "grating" in name:
        return "gratings"
    if "movie" in name:
        return "natural_movies"
    if name.startswith("natural") or IMAGE_NAME.fullmatch(name):
        return "natural_images"
    return None


# Every dataset's own spelling of a stimulus parameter, mapped to one name.
COLUMNS = {
    "orientation_in_degrees": "orientation",
    "spatial_frequency_in_cycles_per_degree": "spatial_frequency",
    "temporal_frequency_in_hz": "temporal_frequency",
    "temporal_frequency_in_cycles_per_second": "temporal_frequency",
    "image_name": "image_name",
}
# Columns that mark a trial as showing nothing, whatever the dataset calls it.
BLANK_COLUMNS = ("is_blank_sweep", "omitted")


@dataclass
class Adapter:
    """Uniform access to one dataset's sessions."""

    dataset: Dataset
    bin_size: float = 0.05
    standardize: bool = True

    def population(
        self,
        session_id: str,
        region: str | None = SHARED_REGION,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Continuous per-neuron signal: (n_time, n_neurons), timestamps, neuron table."""
        if isinstance(self.dataset, Ephys):
            signal, timestamps, neurons = self._spike_rates(session_id, region)
        elif isinstance(self.dataset, VisualLearning):
            signal, timestamps, neurons = self._multiplane_dff(session_id, region)
        else:
            signal, timestamps, neurons = self._dff(session_id)
        return (zscore(signal) if self.standardize else signal), timestamps, neurons

    def _spike_rates(
        self,
        session_id: str,
        region: str | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Bin every unit's spike train onto one common grid."""
        units = self.dataset.load_units(session_id)
        if region is not None:
            in_region = units[units["ecephys_structure_acronym"] == region]
            units = in_region if not in_region.empty else units

        spike_times = list(units["spike_times"])
        start = min(times.min() for times in spike_times)
        stop = max(times.max() for times in spike_times)
        rates = np.stack(
            [
                average_firing_rate([times], self.bin_size, start, stop)
                for times in spike_times
            ],
            axis=1,
        )
        edges = np.arange(start, stop + self.bin_size, self.bin_size)
        return rates, edges[: len(rates)] + self.bin_size / 2, units

    def _dff(self, session_id: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Single-plane dF/F, on the microscope's own timestamps."""
        dff = self.dataset.load_dff(session_id)
        neurons = pd.DataFrame({"roi_id": dff.columns})
        return dff.to_numpy(), dff.index.to_numpy(), neurons

    def _multiplane_dff(
        self,
        session_id: str,
        region: str | None,
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Planes resampled onto one clock, since the scan mirror interleaves them."""
        planes = self.dataset.load_dff(session_id)
        if region is not None:
            selected = {
                name: dff for name, dff in planes.items() if name.startswith(region)
            }
            planes = selected or planes

        timestamps = next(iter(planes.values())).index.to_numpy()
        traces, neurons = [], []
        for name, dff in planes.items():
            plane_time = dff.index.to_numpy()
            for roi in dff.columns:
                traces.append(np.interp(timestamps, plane_time, dff[roi].to_numpy()))
                neurons.append({"plane": name, "roi_id": roi})
        return np.stack(traces, axis=1), timestamps, pd.DataFrame(neurons)

    def stimulus_families(self, session_id: str) -> dict[str, list[str]]:
        """Map each shared stimulus family to this session's names for it."""
        families: dict[str, list[str]] = {}
        for stimulus_type in self.dataset.stimulus_types(session_id):
            family = stimulus_family(stimulus_type)
            if family is not None:
                families.setdefault(family, []).append(str(stimulus_type))
        return {family: sorted(names) for family, names in families.items()}

    def stimulus_names(self, session_id: str) -> list[str]:
        """Every stimulus this session presented, by its own name and by family."""
        names = [str(name) for name in self.dataset.stimulus_types(session_id)]
        return names + list(self.stimulus_families(session_id))

    def presentations(
        self,
        session_id: str,
        stimulus_type: str | None = None,
    ) -> pd.DataFrame:
        """One row per presentation, with canonical parameter names."""
        table = self._stimulus_table(session_id, stimulus_type)
        table = table.rename(columns=COLUMNS)

        blank = pd.Series(False, index=table.index)
        for column in BLANK_COLUMNS:
            if column in table:
                blank |= table[column].fillna(False).astype(bool)
        if "frame" in table:
            # A negative frame is a blank sweep; a missing one just means this
            # stimulus is not made of frames at all (gratings carry no image).
            frames = pd.to_numeric(table["frame"], errors="coerce")
            blank |= frames < 0
        table["is_blank"] = blank

        if stimulus_type is not None and "stimulus_type" in table:
            names = table["stimulus_type"]
            table = table[
                (names == stimulus_type) | (names.map(stimulus_family) == stimulus_type)
            ]
        return table.reset_index(drop=True)

    def _stimulus_table(
        self,
        session_id: str,
        stimulus_type: str | None,
    ) -> pd.DataFrame:
        """Where a dataset keeps its per-presentation stimulus parameters.

        Ephys and visual learning register presentations as intervals, so
        `load_trials` finds them. Visual-coding ophys keeps parameter tables in
        `nwb.stimulus` and leaves only epoch blocks in intervals - and its
        image stimuli are an IndexSeries of template frames rather than a
        table at all.
        """
        if not isinstance(self.dataset, Ophys):
            return self.dataset.load_trials(session_id)

        nwb = self.dataset.load_nwb(session_id)
        if stimulus_type in nwb.stimulus:
            table = nwb.stimulus[stimulus_type].to_dataframe()
            table["stimulus_type"] = stimulus_type
            return table

        series_name = f"{stimulus_type}_stimulus"
        if series_name in nwb.stimulus:
            series = nwb.stimulus[series_name]
            # Stimuli with nothing to show - spontaneous grey screen - are
            # registered as plain intervals rather than a series of frames.
            if hasattr(series, "to_dataframe"):
                table = series.to_dataframe()
                table["stimulus_type"] = stimulus_type
                return table

            start_time = np.asarray(series.timestamps[:])
            duration = np.median(np.diff(start_time)) if len(start_time) > 1 else np.nan
            return pd.DataFrame(
                {
                    "start_time": start_time,
                    "stop_time": start_time + duration,
                    "frame": np.asarray(series.data[:]),
                    "stimulus_type": stimulus_type,
                },
            )
        return self.dataset.load_trials(session_id)

    def behavior(
        self,
        session_id: str,
        name: str = SHARED_BEHAVIOR,
    ) -> pd.DataFrame:
        """Behavioral measure `name` was recorded in, on its own clock."""
        measures = self.dataset.load_behavior(session_id)
        return measures[name] if isinstance(measures, dict) else measures

    def running_speed(self, session_id: str) -> pd.Series:
        """Running speed as a series, the measure all three datasets share."""
        return self.behavior(session_id)[SHARED_BEHAVIOR]

    def images(
        self,
        session_id: str,
        stimulus_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pictures shown, and which one each presentation showed."""
        trials = self.presentations(session_id, stimulus_type)
        trials = trials[~trials["is_blank"]]
        if "frame" not in trials:
            raise KeyError(
                f"{stimulus_type!r} in {self.dataset.name} has no frame column; "
                "only image stimuli have pictures to show",
            )

        classes, labels = np.unique(
            trials["frame"].to_numpy().astype(int),
            return_inverse=True,
        )
        images = []
        for frame in classes:
            path = IMAGES_PATH / stimulus_type / f"image_{int(frame):03d}.png"
            if not path.exists():
                raise FileNotFoundError(
                    f"No saved image for {stimulus_type!r} frame {frame} at {path}; "
                    "run scripts/natural_images.py to write them",
                )
            images.append(np.asarray(Image.open(path)))
        return np.stack(images), labels


if __name__ == "__main__":
    for dataset in (Ephys(), Ophys(), VisualLearning()):
        adapter = Adapter(dataset)
        session_id = dataset.session_ids()[0]
        signal, timestamps, neurons = adapter.population(session_id)
        trials = adapter.presentations(session_id)
        print(
            f"{dataset.name:16s} {session_id[:24]}  "
            f"signal {signal.shape} over {timestamps[-1] - timestamps[0]:7.0f} s  "
            f"neurons {len(neurons):4d}  presentations {len(trials):6d} "
            f"({int(trials['is_blank'].sum())} blank)  "
            f"behavior {adapter.behavior(session_id).shape}",
        )

        if "natural_scenes" in dataset.stimulus_types(session_id):
            images, labels = adapter.images(session_id, "natural_scenes")
            print(
                f"{'':16s} natural_scenes images {images.shape}, labels {labels.shape}",
            )
