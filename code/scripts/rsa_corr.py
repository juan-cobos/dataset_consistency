from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "rsa" / "ephys"
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "ephys_corr"

# drifting_gratings orientations, in the same order used to build each matrix
ORIENTATIONS = [0, 45, 90, 135, 180, 225, 270, 315]
n_orient = len(ORIENTATIONS)

npy_paths = sorted(RESULTS_DIR.glob("*.npy"))
if not npy_paths:
    raise SystemExit(f"no RSA matrices found in {RESULTS_DIR}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

matrices = np.stack([np.load(p) for p in npy_paths])  # (n_sessions, K, K)
iu = np.triu_indices(n_orient, k=1)
real_vectors = matrices[:, iu[0], iu[1]]  # (n_sessions, 28)
real_corr = np.corrcoef(real_vectors)


n_permutations = 1000
null_corr = np.zeros((n_permutations, *real_corr.shape))
for i in range(n_permutations):
    null_corr[i] = np.corrcoef([np.random.permutation(vec) for vec in real_vectors])

n_pairwise = real_corr.shape[-1]
iu_corr = np.triu_indices(n_pairwise, k=1)
real_corr_vec = real_corr[iu_corr[0], iu_corr[1]]
null_corr_vec = null_corr[:, iu_corr[0], iu_corr[1]].ravel()

# Figure 1: real_corr matrix, normalized to the full correlation range
fig1, ax1 = plt.subplots(figsize=(6, 5))
im = ax1.imshow(real_corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax1.set_title("RSA session-by-session correlation")
ax1.set_xlabel("Session")
ax1.set_ylabel("Session")
fig1.colorbar(im, ax=ax1, label="Correlation")
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "real_corr_matrix.png", dpi=150)

# Figure 2: null vs. real distribution of pairwise correlations
fig2, ax2 = plt.subplots(figsize=(6, 5))
bins = np.linspace(-1, 1, 40)
ax2.hist(
    null_corr_vec,
    bins=bins,
    density=True,
    alpha=0.6,
    color="tab:blue",
    label="Null",
)
ax2.hist(
    real_corr_vec,
    bins=bins,
    density=True,
    alpha=0.6,
    color="tab:orange",
    label="Real",
)
ax2.set_xlabel("Correlation")
ax2.set_ylabel("Density")
ax2.set_title("Null vs. real pairwise correlation distributions")
ax2.legend()
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "null_vs_real_hist.png", dpi=150)

plt.show()
