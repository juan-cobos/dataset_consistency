from collections.abc import Callable, Sequence

import numpy as np

from visual_coding.dataset import Ephys, Ophys


class Compose:
    """Chain several transforms, applying each to neuro in order."""

    def __init__(
        self,
        transforms: Sequence[Callable[..., np.ndarray]],
    ) -> None:
        self.transforms = transforms

    def __call__(self, neuro: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            neuro = transform(neuro)
        return neuro

    def __repr__(self) -> str:
        lines = "\n".join(f"    {t}" for t in self.transforms)
        return f"{self.__class__.__name__}(\n{lines}\n)"


class ZScore:
    """Standardize neuro to zero mean and unit variance."""

    def __call__(self, neuro: np.ndarray) -> np.ndarray:
        neuro = np.asarray(neuro, dtype=float)
        mean = neuro.mean(axis=0, keepdims=True)
        std = neuro.std(axis=0, keepdims=True)
        return (neuro - mean) / np.where(std == 0, 1, std)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AverageFiringRate:
    """Bin ragged per-unit spike times into a population-averaged firing rate (Hz)."""

    def __init__(self, bin_size: float = 0.05) -> None:
        self.bin_size = bin_size

    def __call__(self, spike_times: Sequence[np.ndarray]) -> np.ndarray:
        start = min(st.min() for st in spike_times)
        stop = max(st.max() for st in spike_times)
        edges = np.arange(start, stop + self.bin_size, self.bin_size)

        counts = np.zeros(len(edges) - 1)
        for st in spike_times:
            counts += np.histogram(st, bins=edges)[0]
        return counts / len(spike_times) / self.bin_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(bin_size={self.bin_size})"


if __name__ == "__main__":
    ophys = Ophys()
    dff = ophys.load_dff(ophys.session_ids()[0])
    timestamps = dff.index.to_numpy()
    neuro = Compose([ZScore()])(dff.to_numpy())
    print(neuro.shape, timestamps.shape)

    ephys = Ephys()
    units = ephys.load_units(ephys.session_ids()[0])
    neuro = Compose([AverageFiringRate(), ZScore()])(units["spike_times"])
    print(neuro.shape)
