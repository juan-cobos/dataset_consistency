"""Session-by-session agreement of the orientation RDMs, within and across datasets.

`rsa_orientations_{ephys,ophys}.py` now save two matrices per session, so every
analysis here runs over both variants: the raw cosine similarities, and the
centered ones, where each cell's mean response over the orientations is
subtracted first, so the structure reflects tuning rather than the constant
every orientation shares.

Three comparisons, each with a null in which every session's RDM triangle is
shuffled independently before correlating:
  - ephys x ephys, in filename order
  - ophys x ophys, ordered by Cre line so the block structure is visible
  - ephys x ophys, the cross-dataset matrix, ophys again ordered by Cre line

Outputs land in `scripts/output/rsa_corr`.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visual_coding.dataset import Ophys

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "rsa"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "rsa_corr"

# drifting_gratings orientations, in the same order used to build each matrix
ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
VARIANTS = ["raw", "centered"]
N_PERMUTATIONS = 1000

# Cre lines grouped by population class, then by labelled layer within class.
CRE_ORDER = [
    "Slc17a7-IRES2-Cre",
    "Emx1-IRES-Cre",
    "Cux2-CreERT2",
    "Rorb-IRES2-Cre",
    "Nr5a1-Cre",
    "Scnn1a-Tg3-Cre",
    "Rbp4-Cre_KL100",
    "Tlx3-Cre_PL56",
    "Fezf2-CreER",
    "Ntsr1-Cre_GN220",
    "Sst-IRES-Cre",
    "Pvalb-IRES-Cre",
    "Vip-IRES-Cre",
]

iu = np.triu_indices(len(ORIENTATIONS), k=1)
BINS = np.linspace(-1, 1, 40)

# Sized for a projected slide, not a page: everything a step up from default.
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 13,
        "figure.titlesize": 17,
    },
)


def session_vectors(dataset: str, variant: str) -> tuple[list[str], np.ndarray]:
    """Session names and the upper triangle of their RDMs: (n_sessions, 28).

    A session whose triangle is constant - one cell, or every pair at exactly
    1.0 - correlates with nothing, so those are dropped rather than left as
    NaN rows in the correlation matrix.
    """
    variant_dir = RESULTS_DIR / dataset / variant
    paths = sorted(variant_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"no RSA matrices found in {variant_dir}")

    vectors = np.stack([np.load(p)[iu] for p in paths])
    usable = np.ptp(vectors, axis=1) > 0
    if not usable.all():
        print(f"{dataset}/{variant}: dropping {(~usable).sum()} flat-RDM session(s)")
    names = [p.stem for p, keep in zip(paths, usable, strict=True) if keep]
    return names, vectors[usable]


def cre_sorted(names: list[str]) -> tuple[np.ndarray, pd.Series]:
    """Order that groups ophys sessions by Cre line, and the ordered Cre labels.

    The line is the first field of the genotype; anything outside `CRE_ORDER`
    is bucketed as "other" and sorted last.
    """
    metadata = Ophys().metadata.set_index("name")
    missing = [n for n in names if n not in metadata.index]
    if missing:
        raise SystemExit(
            f"{len(missing)} RSA sessions missing from metadata: {missing[:3]}",
        )

    cre = metadata.loc[names, "genotype"].str.split("/", n=1).str[0]
    cre = cre.where(cre.isin(CRE_ORDER), "other")
    cre = cre.astype(pd.CategoricalDtype([*CRE_ORDER, "other"], ordered=True))

    order = np.argsort(cre.values.codes, kind="stable")
    return order, cre.iloc[order]


def correlate(a: np.ndarray, b: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Correlation matrix and its independent entries, within `a` or a vs. b."""
    if b is None:
        corr = np.corrcoef(a)
        return corr, corr[np.triu_indices(len(corr), k=1)]
    corr = np.corrcoef(a, b)[: len(a), len(a) :]
    return corr, corr.ravel()


def shuffle(vectors: np.ndarray) -> np.ndarray:
    """Permute each session's triangle independently."""
    order = np.random.random(vectors.shape).argsort(axis=1)
    return np.take_along_axis(vectors, order, axis=1)


def permutation_test(
    a: np.ndarray,
    b: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Correlation matrix, its entries, the binned null, and a p-value.

    One-sided permutation test on the mean correlation: the observed mean is
    compared against the distribution of per-permutation null means (not the
    pooled null correlations, which pseudo-replicate non-independent pairs).
    Only the null's histogram counts are kept - with ophys' hundreds of
    sessions the full set of permuted correlations is far too large to hold.
    """
    real_corr, real_vec = correlate(a, b)

    null_means = np.zeros(N_PERMUTATIONS)
    null_counts = np.zeros(len(BINS) - 1)
    for i in range(N_PERMUTATIONS):
        _, null_vec = correlate(shuffle(a), None if b is None else shuffle(b))
        null_means[i] = np.nanmean(null_vec)
        null_counts += np.histogram(null_vec, bins=BINS)[0]

    real_mean = np.nanmean(real_vec)
    p_value = (np.sum(null_means >= real_mean) + 1) / (N_PERMUTATIONS + 1)
    return real_corr, real_vec, null_counts, p_value


def annotate_cre(ax: plt.Axes, cre: pd.Series, axis: str) -> None:
    """Draw Cre line boundaries and labels along one axis of a heatmap."""
    counts = cre.value_counts(sort=False).reindex(cre.cat.categories).dropna()
    counts = counts[counts > 0].astype(int)
    edges = np.concatenate([[0], counts.cumsum().values])
    centers = (edges[:-1] + edges[1:]) / 2 - 0.5
    labels = [f"{name} ({n})" for name, n in counts.items()]

    for edge in edges[1:-1]:
        line = ax.axvline if axis == "x" else ax.axhline
        line(edge - 0.5, color="k", linewidth=0.6)
    if axis == "x":
        ax.set_xticks(centers, labels, rotation=45, ha="right", fontsize=10)
    else:
        ax.set_yticks(centers, labels, fontsize=10)


def plot_matrix(
    corr: np.ndarray,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
    cre_x: pd.Series | None = None,
    cre_y: pd.Series | None = None,
) -> None:
    """Heatmap of a correlation matrix, optionally labelled by Cre line."""
    # Square matrices are drawn square; the ephys x ophys one is wide and short.
    square = corr.shape[0] == corr.shape[1]
    # Bigger type needs a bigger canvas: 13 Cre labels down 500 sessions only
    # stay legible if the axes grow with the font.
    width = 12 if cre_x is not None else 8
    height = width - 1 if square else 6 * corr.shape[0] / corr.shape[1] + 2.5
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(
        corr,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="equal" if square else "auto",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if cre_x is not None:
        annotate_cre(ax, cre_x, "x")
        ax.set_xlabel("")
    if cre_y is not None:
        annotate_cre(ax, cre_y, "y")
        ax.set_ylabel("")
    fig.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hist(
    real_vec: np.ndarray,
    null_counts: np.ndarray,
    p_value: float,
    title: str,
    path: Path,
) -> None:
    """Null vs. real distribution of the correlations."""
    real_mean = np.nanmean(real_vec)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.stairs(
        null_counts / (null_counts.sum() * np.diff(BINS)),
        BINS,
        fill=True,
        alpha=0.6,
        color="tab:blue",
        label="Null",
    )
    ax.hist(
        real_vec,
        bins=BINS,
        density=True,
        alpha=0.6,
        color="tab:orange",
        label="Real",
    )
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.axvline(real_mean, color="tab:orange", linestyle="--", linewidth=1)
    ax.text(
        0.02,
        0.95,
        f"mean = {real_mean:.3f}\np = {p_value:.4f}",
        transform=ax.transAxes,
        va="top",
        fontsize=14,
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(label: str, real_vec: np.ndarray, p_value: float, n: str) -> None:
    print(
        f"{label}: {n}, mean correlation = {np.nanmean(real_vec):.3f}, "
        f"permutation p = {p_value:.4f}",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        _, ephys = session_vectors("ephys", variant)
        ophys_names, ophys = session_vectors("ophys", variant)
        order, cre = cre_sorted(ophys_names)
        ophys = ophys[order]

        # ephys x ephys, in filename order
        corr, real_vec, null_counts, p_value = permutation_test(ephys)
        report(f"ephys/{variant}", real_vec, p_value, f"{len(ephys)} sessions")
        plot_matrix(
            corr,
            f"RSA session-by-session correlation (ephys, {variant})",
            OUTPUT_DIR / f"ephys_corr_matrix_{variant}.png",
            "Ephys session",
            "Ephys session",
        )
        plot_hist(
            real_vec,
            null_counts,
            p_value,
            f"Null vs. real pairwise correlations (ephys, {variant})",
            OUTPUT_DIR / f"ephys_null_vs_real_hist_{variant}.png",
        )

        # ophys x ophys, sorted by Cre line
        corr, real_vec, null_counts, p_value = permutation_test(ophys)
        report(f"ophys/{variant}", real_vec, p_value, f"{len(ophys)} sessions")
        plot_matrix(
            corr,
            f"RSA session correlations by Cre line (ophys, {variant})",
            OUTPUT_DIR / f"ophys_corr_matrix_{variant}_by_cre.png",
            "Ophys session",
            "Ophys session",
            cre_x=cre,
            cre_y=cre,
        )
        plot_hist(
            real_vec,
            null_counts,
            p_value,
            f"Null vs. real pairwise correlations (ophys, {variant})",
            OUTPUT_DIR / f"ophys_null_vs_real_hist_{variant}.png",
        )

        # ephys x ophys, ophys sorted by Cre line
        corr, real_vec, null_counts, p_value = permutation_test(ephys, ophys)
        report(
            f"ephys x ophys/{variant}",
            real_vec,
            p_value,
            f"{len(ephys)} x {len(ophys)} sessions",
        )
        plot_matrix(
            corr,
            f"RSA cross-correlation: ephys vs. ophys by Cre line ({variant})",
            OUTPUT_DIR / f"ephys_ophys_corr_matrix_{variant}_by_cre.png",
            "Ophys session",
            "Ephys session",
            cre_x=cre,
        )
        plot_hist(
            real_vec,
            null_counts,
            p_value,
            f"Null vs. real ephys-ophys correlations ({variant})",
            OUTPUT_DIR / f"ephys_ophys_null_vs_real_hist_{variant}.png",
        )

    print(f"saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
