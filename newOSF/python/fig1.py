"""Port of the 'Figure 1: Get embedding and relevant vectors' section (make_figures_behavsim.m lines 93-171).

Skips panels b1/c1-c4: b1 is a decorative random-subset embedding crop, and
c1-c4 are schematic panels built from `rand()` in the original MATLAB script
(placeholder noise for the paper's illustration, not derived from real data).
Ports the two data-derived panels: the sorted similarity matrix and the object montage.
"""
import matplotlib.pyplot as plt
import numpy as np

from helpers import clustering_algorithm


def similarity_plot(data, ax=None):
    ind, _ = clustering_algorithm(3, 5, data.spose_embedding49)
    ordered = data.spose_sim[np.ix_(ind, ind)]
    subset = ordered[::10, ::10]  # matches MATLAB's ind(1:10:end)
    ax = ax or plt.gca()
    im = ax.imshow(subset, vmin=0, vmax=0.9, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Similarity matrix (objects sorted by dominant dimensions)")
    return ax, im, ind


def object_montage(data, ax=None):
    n_rows, n_cols, tile = 30, 61, 150
    canvas = np.zeros((n_rows * tile, n_cols * tile, 3))
    for cnt, img in enumerate(data.im[: n_rows * n_cols]):
        r, c = divmod(cnt, n_cols)
        canvas[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile, :] = img.astype(float) / 255
    ax = ax or plt.gca()
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("All 1,854 THINGS reference images")
    return ax


if __name__ == "__main__":
    from data_loading import load_all

    data = load_all()
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    similarity_plot(data, axes)
    fig.savefig("/tmp/fig1_similarity.png", dpi=120, bbox_inches="tight")
    print("saved /tmp/fig1_similarity.png")

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 7))
    object_montage(data, ax2)
    fig2.savefig("/tmp/fig1_montage.png", dpi=80, bbox_inches="tight")
    print("saved /tmp/fig1_montage.png")
