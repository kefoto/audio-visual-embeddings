"""Port of 'Figure 2: Predict behavior and similarity' (make_figures_behavsim.m lines 182-373)."""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from helpers import squareformq


def behavioral_prediction(data, rng_seed=42):
    """Predict each test triplet's chosen pair from dot-product similarity; per-object accuracy + 95% CI."""
    trip = data.triplet_testdata49
    dp = data.dot_product49
    rng = np.random.default_rng(rng_seed)

    sims = np.stack(
        [dp[trip[:, 0], trip[:, 1]], dp[trip[:, 0], trip[:, 2]], dp[trip[:, 1], trip[:, 2]]], axis=1
    )
    m = sims.max(axis=1, keepdims=True)
    is_max = sims == m
    n_ties = is_max.sum(axis=1)
    choice = is_max.argmax(axis=1)
    tie_rows = np.where(n_ties > 1)[0]
    for r in tie_rows:
        candidates = np.where(is_max[r])[0]
        choice[r] = rng.choice(candidates)

    correct = choice == 0
    behav_predict_acc = 100 * correct.mean()

    n_obj = dp.shape[0]
    behav_predict_obj = np.full(n_obj, np.nan)
    for i_obj in range(n_obj):
        mask = np.any(trip == i_obj, axis=1)
        if mask.any():
            behav_predict_obj[i_obj] = 100 * correct[mask].mean()

    ci95 = 1.96 * np.nanstd(behav_predict_obj) / np.sqrt(n_obj)
    return dict(acc=behav_predict_acc, obj_acc=behav_predict_obj, ci95=ci95, correct=correct)


def noise_ceiling(csv_path):
    """Internal-consistency noise ceiling from repeated triplet presentations."""
    raw = np.loadtxt(csv_path)
    items = raw[:, :3].astype(int) - 1
    choice_col = raw[:, 3].astype(int) - 1  # 0-indexed column (0,1,2) that was chosen

    order = np.argsort(items, axis=1)
    sorted_items = np.take_along_axis(items, order, axis=1)
    new_choice = (order == choice_col[:, None]).argmax(axis=1)

    groups = defaultdict(list)
    for trip, ch in zip(map(tuple, sorted_items), new_choice):
        groups[trip].append(ch)

    consistency = np.array([np.bincount(v, minlength=3).max() / len(v) for v in groups.values()])
    nc = 100 * consistency.mean()
    nc_ci95 = 1.96 * 100 * consistency.std(ddof=0) / np.sqrt(len(consistency))
    return nc, nc_ci95, len(groups)


def plot_accuracy_bar(behav_acc, behav_ci95, nc, nc_ci95, ax=None):
    ax = ax or plt.gca()
    ax.fill_between([0.5, 1.5], nc - nc_ci95, nc + nc_ci95, color=(0.7, 0.7, 0.7))
    ax.bar([1], [behav_acc], width=0.6, color="black")
    ax.errorbar([1], [behav_acc], yerr=behav_ci95, color="black", linewidth=3)
    ax.axhline(100 / 3, color="red", linewidth=3)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(30, 75)
    ax.set_xticks([])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Test-set accuracy vs. noise ceiling")
    return ax


def similarity_comparison(data):
    """Odd-one-out similarity restricted to the 48-object RDM subset (distractor pool = the
    same 48 objects, matching how RDM48_triplet was collected), compared against measured
    RDM48_triplet. Uses embedding2sim_restricted, NOT the general embedding2sim - the RDM48
    ground truth was collected with a 48-item-only odd-one-out task, so the distractor pool
    for the model's prediction must be restricted the same way (see make_figures_behavsim.m
    lines 296-317, which is a different computation from embedding2sim.m).
    """
    from helpers import embedding2sim_restricted

    wp = np.array(data.wordposition48)
    spose_sim48 = embedding2sim_restricted(data.spose_embedding49, wp, wp, verbose=False)

    r48 = np.corrcoef(squareformq(spose_sim48), squareformq(1 - data.RDM48_triplet))[0, 1]

    rng = np.random.default_rng(2)
    c1 = squareformq(spose_sim48)
    c2 = squareformq(1 - data.RDM48_triplet)
    n_pairs = len(c1)
    boot = np.empty(1000)
    for b in range(1000):
        idx = rng.integers(0, n_pairs, n_pairs)
        boot[b] = np.corrcoef(c1[idx], c2[idx])[0, 1]
    z = np.arctanh(r48)
    zci = 1.96 * np.std(np.arctanh(boot))
    ci_lower, ci_upper = np.tanh(z - zci), np.tanh(z + zci)

    reliability48 = np.corrcoef(
        squareformq(1 - data.RDM48_triplet_split1), squareformq(1 - data.RDM48_triplet_split2)
    )[0, 1]
    sh1 = np.corrcoef(squareformq(1 - data.RDM48_triplet_split1), squareformq(spose_sim48))[0, 1]
    sh2 = np.corrcoef(squareformq(1 - data.RDM48_triplet_split2), squareformq(spose_sim48))[0, 1]
    splithalf48 = np.tanh(np.mean(np.arctanh([sh1, sh2])))
    variance_explained48 = splithalf48**2 / reliability48**2

    return dict(
        spose_sim48=spose_sim48,
        r48=r48,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        reliability48=reliability48,
        splithalf48=splithalf48,
        variance_explained48=variance_explained48,
    )


def plot_similarity_panels(sim_result, rdm48, axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(sim_result["spose_sim48"], vmin=0, vmax=1, cmap="viridis")
    axes[0].set_title("predicted similarity matrix")
    axes[0].axis("off")

    axes[1].imshow(1 - rdm48, vmin=0, vmax=1, cmap="viridis")
    axes[1].set_title("measured similarity matrix")
    axes[1].axis("off")

    x = squareformq(sim_result["spose_sim48"])
    y = squareformq(1 - rdm48)
    axes[2].scatter(x, y, s=6, color=(0.5, 0.5, 0.5))
    axes[2].plot([0, 1], [0, 1], "k")
    axes[2].set_xlabel("predicted similarity")
    axes[2].set_ylabel("measured similarity")
    axes[2].legend([f"R = {sim_result['r48']:.2f}"], loc="upper left")
    return axes


if __name__ == "__main__":
    from data_loading import BASE_DIR, load_all

    data = load_all()

    bp = behavioral_prediction(data)
    print(f"Accuracy on test data: {np.nanmean(bp['obj_acc']):.2f} (95% CI across objects: {bp['ci95']:.2f})")

    nc, nc_ci95, n_groups = noise_ceiling(BASE_DIR / "data" / "triplets_noiseceiling.csv")
    print(f"Noise ceiling: {nc:.2f} (95% CI across objects: {nc_ci95:.2f}), n_unique_triplets={n_groups}")

    pct_perf = 100 * (np.nanmean(bp["obj_acc"]) - 100 / 3) / (nc - 100 / 3)
    print(f"Percent performance achieved (subtracting chance): {pct_perf:.2f}")

    fig, ax = plt.subplots(figsize=(3, 6))
    plot_accuracy_bar(np.nanmean(bp["obj_acc"]), bp["ci95"], nc, nc_ci95, ax)
    fig.savefig("/tmp/fig2a.png", dpi=120, bbox_inches="tight")
    print("saved /tmp/fig2a.png")

    print("Running embedding2sim on the full 1854x1854 embedding (~15-20s)...")
    sim_result = similarity_comparison(data)
    print(f"r48 = {sim_result['r48']:.3f} [{sim_result['ci_lower']:.3f}, {sim_result['ci_upper']:.3f}]")
    print(f"reliability48 = {sim_result['reliability48']:.3f}, splithalf48 = {sim_result['splithalf48']:.3f}")
    print(f"variance_explained48 = {sim_result['variance_explained48']:.3f}")

    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_similarity_panels(sim_result, data.RDM48_triplet, axes)
    fig2.savefig("/tmp/fig2b.png", dpi=120, bbox_inches="tight")
    print("saved /tmp/fig2b.png")
