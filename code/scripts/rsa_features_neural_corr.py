from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rsa_features_blocks import block_names, session_matrices
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "rsa"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "rsa_corr"

ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
DATASETS = {"ephys": "#4C72B0", "ophys": "#DD8452"}
FEATURE_DIR = ROOT / "results" / "features" / "ephys"
# feature directory under FEATURE_DIR -> what block_0 is for that trunk
MODELS = {"dinov2-base": "embedding output", "resnet-50": "stem output"}
VARIANTS = ["raw", "centered"]

iu = np.triu_indices(len(ORIENTATIONS), k=1)


def model_reference(model: str) -> dict[str, dict[str, np.ndarray]]:
    """Per-block model RDMs, raw and centered, averaged over the sessions."""
    feature_dir = FEATURE_DIR / model
    paths = sorted(feature_dir.glob("*.npz"))
    if not paths:
        raise SystemExit(f"no feature files in {feature_dir}")

    per_variant: dict[str, dict[str, list[np.ndarray]]] = {v: {} for v in VARIANTS}
    for path in paths:
        raw, centered = session_matrices(path)
        for variant, matrices in (("raw", raw), ("centered", centered)):
            for block, matrix in matrices.items():
                per_variant[variant].setdefault(block, []).append(matrix)

    reference = {
        variant: {block: np.mean(mats, axis=0) for block, mats in per_block.items()}
        for variant, per_block in per_variant.items()
    }
    print(
        f"{model}: {len(paths)} sessions, {len(block_names(reference['raw']))} blocks",
    )
    return reference


def neural_vectors(dataset: str, variant: str) -> np.ndarray:
    """Upper triangle of every session's RDM: (n_sessions, n_conditions)."""
    variant_dir = RESULTS_DIR / dataset / variant
    paths = sorted(variant_dir.glob("*.npy"))
    if not paths:
        raise SystemExit(f"no RSA matrices in {variant_dir}")

    vectors = np.stack([np.load(p)[iu] for p in paths])
    usable = np.ptp(vectors, axis=1) > 0
    if not usable.all():
        print(f"{dataset}/{variant}: dropping {(~usable).sum()} flat-RDM session(s)")
    return vectors[usable]


def correlate(model: dict[str, np.ndarray], vectors: np.ndarray) -> np.ndarray:
    """Spearman of every session against every block: (n_sessions, n_blocks)."""
    blocks = block_names(model)
    return np.array(
        [
            [spearmanr(model[block][iu], session).statistic for block in blocks]
            for session in vectors
        ],
    )


def plot_panel(ax: plt.Axes, depth: list[int], results: dict[str, np.ndarray]) -> None:
    """Correlation against block depth, one colour-coded line per dataset."""
    ax.axhline(0, color="0.6", lw=0.8)
    for dataset, correlations in results.items():
        mean = correlations.mean(axis=0)
        sem = correlations.std(axis=0, ddof=1) / np.sqrt(len(correlations))
        color = DATASETS[dataset]
        ax.fill_between(depth, mean - sem, mean + sem, color=color, alpha=0.25)
        ax.plot(
            depth,
            mean,
            color=color,
            lw=2,
            marker="o",
            ms=5,
            label=f"{dataset} (n={len(correlations)})",
        )
        best = int(np.argmax(mean))
        ax.annotate(
            f"block {depth[best]}, {mean[best]:.2f}",
            (depth[best], mean[best]),
            textcoords="offset points",
            xytext=(6, 8),
            fontsize=8,
            color=color,
        )

    ax.set_xticks(depth)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def plot_depth(
    depth: dict[str, list[int]],
    results: dict[tuple[str, str], dict[str, np.ndarray]],
) -> None:
    """One row per model, one column per variant, sharing the colour code and y axis."""
    fig, axes = plt.subplots(
        len(MODELS),
        len(VARIANTS),
        figsize=(6 * len(VARIANTS), 4.2 * len(MODELS)),
        sharey=True,
    )
    for row, model in enumerate(MODELS):
        for column, variant in enumerate(VARIANTS):
            ax = axes[row, column]
            plot_panel(ax, depth[model], results[model, variant])
            ax.set_title(f"{model} - {variant} RDMs")
            ax.set_xlabel(f"{model} block (0 = {MODELS[model]})")
        axes[row, 0].set_ylabel("Spearman rho with neural RDM")

    fig.suptitle("Model-brain RSA across depth - drifting gratings orientations")
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "rsa_features_neural_corr.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"saved {png_path}")


def main() -> None:
    neural = {
        (dataset, variant): neural_vectors(dataset, variant)
        for dataset in DATASETS
        for variant in VARIANTS
    }

    depth: dict[str, list[int]] = {}
    results: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    saved: dict[str, np.ndarray] = {}
    for model in MODELS:
        reference = model_reference(model)
        blocks = block_names(reference["raw"])
        depth[model] = [int(block.split("_")[1]) for block in blocks]
        saved[f"{model}_blocks"] = np.array(blocks)

        for variant in VARIANTS:
            results[model, variant] = {
                dataset: correlate(reference[variant], neural[dataset, variant])
                for dataset in DATASETS
            }
            for dataset, correlations in results[model, variant].items():
                saved[f"{model}_{variant}_{dataset}"] = correlations
                print(f"{model}/{variant}/{dataset}: {len(correlations)} sessions")
                for i, block in enumerate(blocks):
                    column = correlations[:, i]
                    print(
                        f"  {block:9s} rho {column.mean():+.3f} "
                        f"+/- {column.std(ddof=1):.3f}",
                    )
    plot_depth(depth, results)


if __name__ == "__main__":
    main()
