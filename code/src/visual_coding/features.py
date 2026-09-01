"""Pretrained-model features for the stimuli a session presented."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from visual_coding.analysis import align_epochs, zscore
from visual_coding.dataset import Ophys


class FeatureExtractor:
    """Frozen-model features for `stimulus_type`, aligned to the session's responses.

    `blocks` selects which hidden states to keep - index 0 is the embedding
    output and the last is the final block. Which depth best predicts a visual
    area is an empirical question, so several are kept by default.
    """

    def __init__(
        self,
        dataset: Ophys,
        session_id: str,
        model_name: str = "facebook/dinov2-base",
        stimulus_type: str = "natural_scenes",
        blocks: tuple[int, ...] = (3, 6, 9, 12),
        window: tuple[float, float] = (0.0, 1.0),
        bin_size: float = 0.25,
    ) -> None:
        self.dataset = dataset
        self.session_id = session_id
        self.model_name = model_name
        self.stimulus_type = stimulus_type
        self.blocks = blocks
        self.window = window
        self.bin_size = bin_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Classes follow np.unique, the convention `Decoder` uses for its
        # labels, so class k means the same image on both sides.
        series = dataset.load_stimulus(session_id, stimulus_type)
        self.classes, self.labels = np.unique(
            np.asarray(series.data[:]),
            return_inverse=True,
        )
        self.event_times = np.asarray(series.timestamps[:])

        templates = dataset.load_nwb(session_id).stimulus_template
        name = (
            stimulus_type if stimulus_type in templates else f"{stimulus_type}_template"
        )
        self.images = np.asarray(templates[name].data[:])[self.classes]

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    def features(self) -> dict[str, np.ndarray]:
        """Features of each unique image: {block: (n_images, n_features)}."""
        # The trunks are trained on RGB, so the single grayscale channel is
        # replicated; the processor does the resize, crop and normalization.
        rgb = [np.repeat(image[:, :, None], 3, axis=2) for image in self.images]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            hidden = self.model(**inputs, output_hidden_states=True).hidden_states
        # One row per image: the block's tokens averaged together.
        return {
            f"block_{i}": hidden[i].mean(dim=1).float().cpu().numpy()
            for i in self.blocks
        }

    def epochs(self) -> np.ndarray:
        """Per-ROI response to each presentation: (n_presentations, n_rois, n_bins)."""
        dff = self.dataset.load_dff(self.session_id)
        traces = zscore(dff.to_numpy())
        timestamps = dff.index.to_numpy()
        aligned = [
            align_epochs(
                trace,
                timestamps,
                self.event_times,
                self.window,
                self.bin_size,
            )[0]
            for trace in traces.T
        ]
        return np.stack(aligned, axis=1)

    def extract(
        self,
        block: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Features and epoch traces per presentation, in the order they were shown."""
        block = f"block_{self.blocks[-1]}" if block is None else block
        features = self.features()
        if block not in features:
            raise KeyError(
                f"{block!r} was not extracted; available: {sorted(features)}",
            )

        responses = self.epochs()
        keep = ~np.isnan(responses).any(axis=(1, 2))
        return features[block][self.labels][keep], responses[keep], self.labels[keep]


if __name__ == "__main__":
    ophys = Ophys()
    session_id = next(
        sid
        for sid in ophys.session_ids()
        if "natural_scenes" in ophys.stimulus_types(sid)
    )
    extractor = FeatureExtractor(ophys, session_id)
    print(
        f"{session_id}: {len(extractor.classes)} images, {len(extractor.labels)} presentations",
    )

    for block, matrix in extractor.features().items():
        print(f"  {block:8s} {matrix.shape}")

    features, responses, labels = extractor.extract()
    print(
        f"features {features.shape}  responses {responses.shape}  labels {labels.shape}",
    )
