from __future__ import annotations

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from visual_coding.adapter import SHARED_REGION, Adapter
from visual_coding.analysis import align_epochs
from visual_coding.dataset import Dataset


class FeatureExtractor:
    """Frozen-model features for `stimulus_type`, aligned to the session's responses.

    `blocks` selects which hidden states to keep - index 0 is the embedding
    output and the last is the final block. Which depth best predicts a visual
    area is an empirical question, so several are kept by default.
    """

    def __init__(
        self,
        dataset: Dataset,
        session_id: str,
        model_name: str = "facebook/dinov2-base",
        stimulus_type: str = "natural_scenes",
        blocks: tuple[int, ...] = (3, 6, 9, 12),
        window: tuple[float, float] = (0.0, 1.0),
        bin_size: float = 0.25,
        region: str = SHARED_REGION,
    ) -> None:
        self.dataset = dataset
        self.session_id = session_id
        self.model_name = model_name
        self.stimulus_type = stimulus_type
        self.blocks = blocks
        self.window = window
        self.bin_size = bin_size
        self.region = region
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = Adapter(dataset, bin_size=bin_size)

        trials = self.adapter.presentations(session_id, stimulus_type)
        self.trials = trials[~trials["is_blank"]]
        self.event_times = self.trials["start_time"].to_numpy()
        self.images, self.labels = self.adapter.images(session_id, stimulus_type)

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
        """Per-neuron response to each presentation: (n_presentations, n_neurons, n_bins).

        Spike trains arrive binned into firing rates and dF/F z-scored, so the
        adapter hands back one continuous signal either way.
        """
        signal, timestamps, _ = self.adapter.population(
            self.session_id,
            region=self.region,
        )
        aligned = [
            align_epochs(
                trace,
                timestamps,
                self.event_times,
                self.window,
                self.bin_size,
            )[0]
            for trace in signal.T
        ]
        return np.stack(aligned, axis=1)

    def extract(
        self,
        block: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Features and epoch traces per presentation, in the order they were shown.

        Returns (features, responses, labels): features is
        (n_presentations, n_features) for `block` - the last one by default -
        responses is (n_presentations, n_neurons, n_bins), and labels says
        which image each row showed. Presentations whose window runs off the
        end of the recording are dropped from all three.
        """
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
    from visual_coding.dataset import Ephys, Ophys

    for dataset in (Ophys(), Ephys()):
        session_id = next(
            sid
            for sid in dataset.session_ids()
            if "natural_scenes" in dataset.stimulus_types(sid)
        )
        extractor = FeatureExtractor(dataset, session_id)
        features, responses, labels = extractor.extract()
        print(
            f"{dataset.name} {session_id[:24]}: {len(extractor.images)} images, "
            f"features {features.shape}, responses {responses.shape}",
        )
