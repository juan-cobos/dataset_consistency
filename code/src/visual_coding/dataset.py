from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb
from PIL import Image

from visual_coding.utils import resolve_capsule_path

ROOT = resolve_capsule_path("data")
NEUROPIXELS_PATH = ROOT / "visual_coding_neuropixels"
OPHYS_PATH = ROOT / "visual_coding_ophys"
VISUAL_LEARNING_PATH = ROOT / "Visual-Learning-SWDB"
METADATA_PATH = ROOT / "metadata"


@dataclass
class Dataset:
    """Common session lookup/loading shared by the ephys and ophys datasets."""

    name: str
    data_path: Path
    metadata_path: Path

    @cached_property
    def metadata(self) -> pd.DataFrame:
        """Load the per-session metadata table for this dataset."""
        return pd.read_csv(self.metadata_path)

    @cached_property
    def sessions(self) -> dict[str, Path]:
        """Map each session_id to its data directory."""
        return {p.name: p for p in sorted(self.data_path.iterdir()) if p.is_dir()}

    def session_ids(self) -> list[str]:
        """List all available session_ids."""
        return list(self.sessions)

    def session_metadata(self, session_id: str) -> pd.Series:
        """Return the metadata row for session_id."""
        rows = self.metadata.loc[self.metadata["name"] == session_id]
        if rows.empty:
            raise KeyError(f"No metadata found for session {session_id!r}")
        return rows.iloc[0]

    def load_nwb(self, session_id: str) -> pynwb.NWBFile:
        """Load the NWB file for session_id."""
        session_dir = self.sessions[session_id]
        nwb_path = next(session_dir.glob("*.nwb.zarr"))
        return pynwb.read_nwb(nwb_path)

    def start_time(self, session_id: str) -> datetime:
        """Return the start time for session_id."""
        return self.load_nwb(session_id).session_start_time

    def load_trials(self, session_id: str) -> pd.DataFrame:
        """Load all stimulus presentation intervals for session_id, tagged by stimulus_type."""
        nwb = self.load_nwb(session_id)
        if nwb.trials is not None:
            return nwb.trials.to_dataframe()

        frames = []
        for name, table in (nwb.intervals or {}).items():
            if name == "invalid_times":
                continue
            df = table.to_dataframe()
            if "stimulus_type" not in df.columns:
                df["stimulus_type"] = (
                    df["stimulus_name"] if "stimulus_name" in df.columns else name
                )
            frames.append(df)
        return pd.concat(frames) if frames else pd.DataFrame()

    def stimulus_types(self, session_id: str) -> list:
        """Return the unique stimulus types presented in session_id."""
        trials = self.load_trials(session_id)
        return trials.stimulus_type.unique().tolist()


@dataclass
class Ephys(Dataset):
    name: str = "ephys"
    data_path: Path = NEUROPIXELS_PATH
    metadata_path: Path = METADATA_PATH / "visual_coding_neuropixels_metadata.csv"
    amplitude_cutoff: float = 0.1
    presence_ratio: float = 0.95
    isi_violations: float = 0.5

    def load_units(self, session_id: str, quality_only: bool = True) -> pd.DataFrame:
        """Load the recorded units (neurons) and their properties for session_id."""
        units = self.load_nwb(session_id).units.to_dataframe()
        if not quality_only:
            return units
        return units[
            (units["quality"] == "good")
            & (units["amplitude_cutoff"] <= self.amplitude_cutoff)
            & (units["presence_ratio"] >= self.presence_ratio)
            & (units["isi_violations"] <= self.isi_violations)
            & (units["ecephys_structure_acronym"] != "")
        ]

    def brain_structures(self, session_id: str) -> np.ndarray:
        """Return the unique brain structures recorded across units in session_id."""
        # Probe coverage, so every sorted unit counts, not just the good ones.
        acronyms = self.load_units(session_id, quality_only=False)[
            "ecephys_structure_acronym"
        ]
        return acronyms[acronyms != ""].unique()

    def load_behavior(self, session_id: str) -> dict[str, pd.DataFrame]:
        """Load running-wheel behavioral time series for session_id."""
        running = self.load_nwb(session_id).processing["running"]
        return {
            name: pd.DataFrame({name: ts.data[:]}, index=ts.timestamps[:])
            for name, ts in running.data_interfaces.items()
        }


@dataclass
class Ophys(Dataset):
    name: str = "ophys"
    data_path: Path = OPHYS_PATH
    metadata_path: Path = METADATA_PATH / "visual_coding_ophys_metadata.csv"

    def _stimulus_template(
        self,
        nwb: pynwb.NWBFile,
        stimulus_type: str,
    ) -> pynwb.image.ImageSeries:
        """Return the template image series for stimulus_type."""
        templates = nwb.stimulus_template
        for name in (stimulus_type, f"{stimulus_type}_template"):
            if name in templates:
                return templates[name]
        raise KeyError(
            f"No stimulus template for {stimulus_type!r}; available: {list(templates.keys())}",
        )

    def load_stimulus(
        self,
        session_id: str,
        stimulus_type: str,
    ) -> pynwb.image.IndexSeries:
        """Return the presentation series (data=frame index, timestamps=presentation times)."""
        nwb = self.load_nwb(session_id)
        self._stimulus_template(nwb, stimulus_type)
        return nwb.stimulus[f"{stimulus_type}_stimulus"]

    def load_image(
        self,
        session_id: str,
        stimulus_type: str,
        frame_idx: int,
    ) -> Image.Image:
        """Return the frame shown at presentation frame_idx of stimulus_type."""
        nwb = self.load_nwb(session_id)
        template = self._stimulus_template(nwb, stimulus_type)
        presentations = nwb.stimulus[f"{stimulus_type}_stimulus"]
        return Image.fromarray(template.data[presentations.data[frame_idx]])

    def brain_structures(self, session_id: str) -> np.ndarray:
        """Return the unique brain structures imaged across planes in session_id."""
        planes = self.load_nwb(session_id).imaging_planes.values()
        return np.unique([plane.location for plane in planes])

    def load_dff(self, session_id: str) -> pd.DataFrame:
        """Load dF/F traces for every ROI in session_id, indexed by timestamp."""
        rrs = (
            self.load_nwb(session_id)
            .processing["ophys"]
            .data_interfaces["DfOverF"]
            .roi_response_series["DfOverF"]
        )
        roi_ids = rrs.rois.table["global_roi_id"][rrs.rois.data[:]]
        return pd.DataFrame(rrs.data[:], index=rrs.timestamps[:], columns=roi_ids)

    def load_behavior(self, session_id: str) -> pd.DataFrame:
        """Load the running speed time series for session_id."""
        rs = (
            self.load_nwb(session_id)
            .processing["behavior"]
            .data_interfaces["BehavioralTimeSeries"]
            .time_series["running_speed"]
        )
        return pd.DataFrame({"running_speed": rs.data[:]}, index=rs.timestamps[:])


@dataclass
class VisualLearning(Dataset):
    """Multiplane-ophys visual-learning sessions (behavioral change-detection task)."""

    name: str = "visual_learning"
    data_path: Path = VISUAL_LEARNING_PATH
    metadata_path: Path = METADATA_PATH / "visual_learning_session_metadata.csv"

    def load_trials(self, session_id: str) -> pd.DataFrame:
        """Load stimulus presentations for session_id, tagged by stimulus_type (image_name).

        `nwb.trials` holds per-trial behavioral outcomes (go/catch/hit/miss/...)
        rather than stimulus identity; use `load_behavior_trials` for that table.
        """
        nwb = self.load_nwb(session_id)
        df = nwb.intervals["stimulus_presentations"].to_dataframe()
        df["stimulus_type"] = df["image_name"]
        return df

    def load_behavior_trials(self, session_id: str) -> pd.DataFrame:
        """Load the per-trial behavioral outcome table (go/catch/hit/miss/...) for session_id."""
        return self.load_nwb(session_id).trials.to_dataframe()

    def brain_structures(self, session_id: str) -> np.ndarray:
        """Return the unique brain structures imaged across planes in session_id."""
        planes = self.load_nwb(session_id).imaging_planes.values()
        return np.unique([plane.location.split()[1] for plane in planes])

    def plane_names(self, session_id: str) -> list[str]:
        """Return the names of the imaging-plane processing modules for session_id."""
        return list(self.load_nwb(session_id).imaging_planes.keys())

    def load_dff(
        self,
        session_id: str,
        plane: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Load dF/F traces per imaging plane for session_id, indexed by timestamp.

        Planes are recorded on an interleaved scan mirror, so timestamps differ
        slightly between planes; traces are kept per-plane rather than concatenated.
        Pass `plane` (e.g. "VISp_0") to load a single plane instead of all of them.
        """
        nwb = self.load_nwb(session_id)
        plane_names = [plane] if plane is not None else self.plane_names(session_id)
        traces = {}
        for name in plane_names:
            rrs = (
                nwb.processing[name]
                .data_interfaces["dff_timeseries"]
                .roi_response_series["dff_timeseries"]
            )
            traces[name] = pd.DataFrame(
                rrs.data[:],
                index=rrs.timestamps[:],
                columns=rrs.rois.data[:],
            )
        return traces

    def load_behavior(self, session_id: str) -> pd.DataFrame:
        """Load the running speed time series for session_id."""
        rs = self.load_nwb(session_id).processing["running"].data_interfaces["speed"]
        return pd.DataFrame({"running_speed": rs.data[:]}, index=rs.timestamps[:])


ALL_DATASETS = {"ophys": Ophys, "ephys": Ephys, "visual_learning": VisualLearning}

if __name__ == "__main__":
    ophys = Ophys()
    session_id = ophys.session_ids()[0]
    print(ophys.load_dff(session_id).head())

    presentations = ophys.load_stimulus(session_id, "natural_movie_one")
    print(f"natural_movie_one presentation 0 @ t={presentations.timestamps[0]:.3f}s")
    ophys.load_image(session_id, "natural_movie_one", 0).show()
