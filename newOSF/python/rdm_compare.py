"""Build the full 1854x1854 RDM from a retrained embedding and compare it to the paper's
published RDM (data/spose_similarity.mat), using the same odd-one-out similarity metric
(embedding2sim) and correlation convention as Figure 2's RDM48 comparison.
"""
import matplotlib.pyplot as plt
import numpy as np

from helpers import clustering_algorithm, embedding2sim, squareformq


def compute_rdm(embedding, verbose=True):
    """dissimilarity = 1 - odd-one-out similarity, full n_objects x n_objects."""
    sim = embedding2sim(embedding, verbose=verbose)
    return 1 - sim


def compare_rdms(dissim_a, dissim_b, n_boot=1000, rng_seed=2):
    va = squareformq(dissim_a)
    vb = squareformq(dissim_b)
    r = np.corrcoef(va, vb)[0, 1]

    rng = np.random.default_rng(rng_seed)
    n_pairs = len(va)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_pairs, n_pairs)
        boot[b] = np.corrcoef(va[idx], vb[idx])[0, 1]
    z = np.arctanh(r)
    zci = 1.96 * np.std(np.arctanh(boot))
    return dict(r=r, ci_lower=np.tanh(z - zci), ci_upper=np.tanh(z + zci))


def plot_rdm_comparison(dissim_paper, dissim_mine, order, label_mine="retrained (54D)", axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 5))

    sim_paper_ord = 1 - dissim_paper[np.ix_(order, order)]
    sim_mine_ord = 1 - dissim_mine[np.ix_(order, order)]
    step = 10  # subsample for a readable/faster heatmap, matches Figure 1's ind(1:10:end)

    axes[0].imshow(sim_paper_ord[::step, ::step], vmin=0, vmax=0.9, cmap="viridis")
    axes[0].set_title("published RDM (49D)")
    axes[0].axis("off")

    axes[1].imshow(sim_mine_ord[::step, ::step], vmin=0, vmax=0.9, cmap="viridis")
    axes[1].set_title(f"{label_mine} RDM")
    axes[1].axis("off")

    x = squareformq(dissim_paper)
    y = squareformq(dissim_mine)
    axes[2].scatter(x, y, s=1, alpha=0.15, color=(0.3, 0.3, 0.3))
    axes[2].plot([0, 1], [0, 1], "k", linewidth=1)
    axes[2].set_xlabel("published dissimilarity (49D)")
    axes[2].set_ylabel(f"{label_mine} dissimilarity")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    return axes


if __name__ == "__main__":
    from pathlib import Path

    from data_loading import BASE_DIR, load_all

    data = load_all()
    dissim_paper = data.dissim  # already loaded from data/spose_similarity.mat

    W_mine = np.load(
        BASE_DIR.parent / "results" / "behavioral" / "90d" / "0.008" / "seed42" / "weights_sorted.npy"
    )
    print(f"Loaded retrained embedding: {W_mine.shape}")

    print("Computing full 1854x1854 RDM from the 54D retrained embedding (~15-20s)...")
    dissim_mine = compute_rdm(W_mine)

    out_dir = BASE_DIR.parent / "results" / "behavioral" / "90d" / "0.008" / "seed42"
    np.save(out_dir / "rdm_54d.npy", dissim_mine)
    print(f"Saved RDM to {out_dir / 'rdm_54d.npy'}")

    stats = compare_rdms(dissim_paper, dissim_mine)
    print(f"RDM correlation: r = {stats['r']:.3f}  [{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]")

    ind, _ = clustering_algorithm(3, 5, data.spose_embedding49)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_rdm_comparison(dissim_paper, dissim_mine, ind, axes=axes)
    axes[2].legend([f"R = {stats['r']:.3f}"], loc="upper left")
    fig.suptitle("Published 49D RDM vs. retrained 54D RDM (full 1854-object odd-one-out similarity)")
    fig.savefig("/tmp/rdm_comparison.png", dpi=120, bbox_inches="tight")
    print("saved /tmp/rdm_comparison.png")
