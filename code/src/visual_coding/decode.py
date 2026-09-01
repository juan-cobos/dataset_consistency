"""Cross-validated stimulus decoding, shared across the ephys and ophys datasets.

`Decoder` wraps one session of any `Dataset` and exposes the same pipeline for
all of them: pull the per-presentation stimulus table, bin the neural signal
(spike times for ephys, dF/F traces for the imaging datasets) and the running
speed into the same within-trial bins, then cross-validate a linear SVM on any
combination of those feature sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from visual_coding.analysis import align_epochs, average_firing_rate, zscore
from visual_coding.dataset import (
    SHARED_BEHAVIOR,
    SHARED_REGION,
    Dataset,
    Ephys,
    Ophys,
    stimulus_family,
)

# The ophys stimulus tables spell out the units the ephys tables leave implicit.
OPHYS_COLUMNS = {
    "orientation_in_degrees": "orientation",
    "spatial_frequency_in_cycles_per_degree": "spatial_frequency",
    "temporal_frequency_in_hz": "temporal_frequency",
}


@dataclass
class Decoder:
    """Decode a stimulus parameter from one session of any visual-coding dataset.

    `stimulus_type` selects the presentations to decode: a dataset's own
    stimulus name (`"drifting_gratings"`), a shared family (`"gratings"`, which
    is how visual learning names its stimuli - one per image, `"gratings_270"`),
    or None to keep every trial in the session's stimulus table. `label` names
    the column to decode from them, e.g. grating `orientation` or `image_name`.
    The window defaults to the 2 s a drifting grating is on screen.
    """

    dataset: Dataset
    session_id: str
    stimulus_type: str | None = "drifting_gratings"
    label: str = "orientation"
    region: str = SHARED_REGION
    behavior: str = SHARED_BEHAVIOR
    window: tuple[float, float] = (0.0, 2.0)
    bin_size: float = 0.25
    n_folds: int = 5
    seed: int = 0
    verbose: bool = True

    @cached_property
    def trials(self) -> pd.DataFrame:
        """Per-presentation stimulus table, restricted to trials with a label."""
        trials = self._stimulus_table()
        if self.stimulus_type is not None and "stimulus_type" in trials:
            names = trials["stimulus_type"]
            trials = trials[
                (names == self.stimulus_type)
                | (names.map(stimulus_family) == self.stimulus_type)
            ]
        if self.label not in trials.columns:
            raise KeyError(
                f"{self.label!r} not in the {self.dataset.name} stimulus table; "
                f"available: {sorted(trials.columns)}",
            )
        return trials.dropna(subset=[self.label])

    def _stimulus_table(self) -> pd.DataFrame:
        """Dataset-specific source of the per-presentation stimulus table."""
        if not isinstance(self.dataset, Ophys):
            return self.dataset.load_trials(self.session_id)

        nwb = self.dataset.load_nwb(self.session_id)
        if self.stimulus_type not in nwb.stimulus:
            raise KeyError(
                f"No stimulus table for {self.stimulus_type!r} in session "
                f"{self.session_id}; available: {list(nwb.stimulus)}",
            )
        table = nwb.stimulus[self.stimulus_type].to_dataframe()
        return table.rename(columns=OPHYS_COLUMNS)

    @property
    def event_times(self) -> np.ndarray:
        """Trial start times."""
        return self.trials["start_time"].to_numpy()

    @cached_property
    def y(self) -> np.ndarray:
        """Per-trial label value, e.g. the grating orientation in degrees.

        Trained on directly rather than on class indices, so a model fit on
        one session predicts labels another session can be scored against.
        """
        return self.trials[self.label].to_numpy()

    @property
    def classes(self) -> np.ndarray:
        """The distinct label values being decoded."""
        return np.unique(self.y)

    @property
    def chance(self) -> float:
        """Accuracy expected from guessing (balanced classes)."""
        return 1 / len(self.classes)

    def _align(self, values: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Bin one signal into the trial window: (n_trials, n_bins)."""
        epochs, _ = align_epochs(
            values,
            timestamps,
            self.event_times,
            self.window,
            self.bin_size,
        )
        return epochs

    def _spike_responses(self) -> np.ndarray:
        """Per-unit binned firing rate: (n_units, n_trials, n_bins)."""
        units = self.dataset.load_units(self.session_id)
        in_region = units[units["ecephys_structure_acronym"] == self.region]
        if in_region.empty:
            in_region = units

        start = self.event_times.min() + self.window[0]
        stop = self.event_times.max() + self.window[1]
        edges = np.arange(start, stop + self.bin_size, self.bin_size)
        timestamps = edges[:-1] + self.bin_size / 2

        return np.stack(
            [
                self._align(
                    average_firing_rate([spike_times], self.bin_size, start, stop),
                    timestamps,
                )
                for spike_times in in_region["spike_times"]
            ],
        )

    def _trace_responses(self) -> np.ndarray:
        """Per-ROI binned z-scored dF/F: (n_rois, n_trials, n_bins).

        Visual-learning sessions image several planes on an interleaved scan
        mirror, so each plane is aligned on its own timestamps before the ROIs
        are pooled.
        """
        traces = self.dataset.load_dff(self.session_id)
        if isinstance(traces, pd.DataFrame):
            traces = {self.region: traces}
        else:
            traces = {
                plane: dff
                for plane, dff in traces.items()
                if plane.startswith(self.region)
            } or traces

        responses = [
            self._align(trace, dff.index.to_numpy())
            for dff in traces.values()
            for trace in zscore(dff.to_numpy()).T
        ]
        return np.stack(responses)

    @cached_property
    def neural_features(self) -> np.ndarray:
        """Per-neuron response (n_trials, n_neurons * n_bins)."""
        responses = (
            self._spike_responses()
            if isinstance(self.dataset, Ephys)
            else self._trace_responses()
        )
        n_neurons, n_trials, n_bins = responses.shape
        if self.verbose:
            kind = "units" if isinstance(self.dataset, Ephys) else "ROIs"
            print(f"{n_neurons} {self.region} {kind} x {n_bins} bins")
        return responses.transpose(1, 0, 2).reshape(n_trials, n_neurons * n_bins)

    @cached_property
    def running_features(self) -> np.ndarray:
        """Return running speed: (n_trials, n_bins)."""
        running = self.dataset.load_behavior(self.session_id)
        if isinstance(running, dict):  # ephys returns one frame per running series
            running = running[self.behavior]
        return self._align(running[self.behavior].to_numpy(), running.index.to_numpy())

    def feature_sets(self) -> dict[str, np.ndarray]:
        """Neural, behavioral and combined feature matrices."""
        neural = self.neural_features
        running = self.running_features
        return {
            "neural only": neural,
            "neural + running": np.hstack([neural, running]),
            "running only": running,
        }

    def pipeline(self) -> Pipeline:
        """Standardize, then a linear SVM."""
        return make_pipeline(StandardScaler(), SVC(kernel="linear"))

    @staticmethod
    def valid_rows(X: np.ndarray) -> np.ndarray:
        """Trials whose features are fully sampled (no window running off a trace)."""
        return ~np.isnan(X).any(axis=1)

    def decode(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Cross-validated linear-SVM accuracy, one score per fold."""
        y = self.y if y is None else y
        valid = self.valid_rows(X)
        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.seed,
        )
        return cross_val_score(self.pipeline(), X[valid], y[valid], cv=cv)

    def fit(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        classes: np.ndarray | None = None,
    ) -> Pipeline:
        """Fit the pipeline on this session, optionally on a subset of `classes`."""
        X = self.neural_features if X is None else X
        y = self.y if y is None else y
        keep = self.valid_rows(X)
        if classes is not None:
            keep &= np.isin(y, classes)
        return self.pipeline().fit(X[keep], y[keep])

    def predict(
        self,
        model: Pipeline,
        X: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply a fitted model to this session: (predicted labels, trials kept."""
        X = self.neural_features if X is None else X
        if X.shape[1] != model.n_features_in_:
            raise ValueError(
                f"model expects {model.n_features_in_} features, this one has "
                f"{X.shape[1]}; a model only applies to the population it was fit on",
            )
        keep = self.valid_rows(X)
        return model.predict(X[keep]), keep

    def run(
        self,
        conditions: dict[str, np.ndarray] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Score every feature set: a summary table plus the per-fold scores."""
        conditions = self.feature_sets() if conditions is None else conditions
        if self.verbose:
            print(
                f"=== {self.dataset.name} {self.session_id}: {self.stimulus_type}, "
                f"{len(self.classes)} {self.label}s, window {self.window}, "
                f"{len(self.event_times)} trials ===",
            )

        rows, folds = [], {}
        for name, X in conditions.items():
            n_trials = int((~np.isnan(X).any(axis=1)).sum())
            scores = self.decode(X)
            folds[name] = scores
            rows.append(
                {
                    "condition": name,
                    "n_features": X.shape[1],
                    "n_trials": n_trials,
                    "accuracy": scores.mean(),
                    "std": scores.std(),
                    "x_chance": scores.mean() / self.chance,
                    **{f"fold_{i}": score for i, score in enumerate(scores)},
                },
            )
            if self.verbose:
                print(
                    f"  {name:20s} {X.shape[1]:5d} feats {n_trials:5d} trials  "
                    f"acc {scores.mean():.3f} +/- {scores.std():.3f}"
                    f"  ({scores.mean() / self.chance:.2f}x chance)",
                )
        if self.verbose:
            print(f"\nchance = {self.chance:.3f}")
        return pd.DataFrame(rows), folds
