from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pynwb

ROOT = Path("/root/capsule/data")
try:
    is_capsule = ROOT.exists()
except OSError:
    is_capsule = False
if not is_capsule:
    ROOT = Path(__file__).resolve().parents[3] / "data"
NEUROPIXELS_PATH = ROOT / "visual_coding_neuropixels"
OPHYS_PATH = ROOT / "visual_coding_ophys"
METADATA_PATH = ROOT / "metadata"

DATASETS = ("ophys", "ephys")


@dataclass
class Dataset:
    """Common session lookup/loading shared by the ephys and ophys datasets."""

    name: str
    data_path: Path
    metadata_path: Path

    @cached_property
    def metadata(self) -> pd.DataFrame:
        return pd.read_csv(self.metadata_path)

    @cached_property
    def sessions(self) -> dict[str, Path]:
        return {p.name: p for p in sorted(self.data_path.iterdir()) if p.is_dir()}

    def session_ids(self) -> list[str]:
        return list(self.sessions)

    def session_metadata(self, session_id: str) -> pd.Series:
        rows = self.metadata.loc[self.metadata["name"] == session_id]
        if rows.empty:
            raise KeyError(f"No metadata found for session {session_id!r}")
        return rows.iloc[0]

    def load_nwb(self, session_id: str) -> pynwb.NWBFile:
        session_dir = self.sessions[session_id]
        nwb_path = next(session_dir.glob("*.nwb.zarr"))
        return pynwb.read_nwb(nwb_path)

    def start_time(self, session_id: str) -> datetime:
        """Start time for session_id."""
        return self.load_nwb(session_id).session_start_time

    def load_trials(self, session_id: str) -> pd.DataFrame:
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

    def stimulus_types(self, session_id: str) -> np.ndarray:
        trials = self.load_trials(session_id)
        return trials.stimulus_type.unique()


@dataclass
class Ephys(Dataset):
    name: str = "ephys"
    data_path: Path = NEUROPIXELS_PATH
    metadata_path: Path = METADATA_PATH / "visual_coding_neuropixels_metadata.csv"

    def load_units(self, session_id: str) -> pd.DataFrame:
        return self.load_nwb(session_id).units.to_dataframe()

    def load_unit_timestamps(self, session_id: str) -> pd.Series:
        return self.load_units(session_id)["spike_times"]

    def load_behavior(self, session_id: str) -> dict[str, pd.DataFrame]:
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

    def load_dff(self, session_id: str) -> pd.DataFrame:
        rrs = (
            self.load_nwb(session_id)
            .processing["ophys"]
            .data_interfaces["DfOverF"]
            .roi_response_series["DfOverF"]
        )
        roi_ids = rrs.rois.table["global_roi_id"][rrs.rois.data[:]]
        return pd.DataFrame(rrs.data[:], index=rrs.timestamps[:], columns=roi_ids)

    def load_behavior(self, session_id: str) -> pd.DataFrame:
        rs = (
            self.load_nwb(session_id)
            .processing["behavior"]
            .data_interfaces["BehavioralTimeSeries"]
            .time_series["running_speed"]
        )
        return pd.DataFrame({"running_speed": rs.data[:]}, index=rs.timestamps[:])


if __name__ == "__main__":
    ophys = Ophys()
    session_id = ophys.session_ids()[0]
    print(ophys.load_dff(session_id).head())
