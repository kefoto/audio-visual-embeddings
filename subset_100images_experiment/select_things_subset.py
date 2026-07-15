"""
Select a 100-image subset of THINGS that preserves the representational
validity established in Hebart et al. (2020, Nat Hum Behav).

Expected inputs (from the THINGS/SPoSE OSF release, https://osf.io/z2784/):
  spose_embedding_49d.txt   -- 1854 x 49 matrix, one row per THINGS object,
                               tab/space separated floats (non-negative dims)
  category_labels.csv       -- 1854 rows, column 'category' = superordinate
                               category name (from THINGS 1.0 taxonomy)
  triplet_dataset.csv       -- columns [i, j, k, choice] where choice in {i,j,k}
                               is the index of the item chosen as the ODD ONE OUT
                               (i.e. NOT part of the most-similar pair)

All indices below are 0-based and refer to rows of the embedding matrix.

Subset selection works in a 2D MDS projection of the 49D SPoSE embedding:
each category's items are projected to 2D (preserving pairwise dissimilarity),
K-means finds cluster centroids in that 2D space, and nearest-neighbor lookup
snaps each centroid back to a real item.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "things_data"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"

RNG_SEED = 0


# ---------------------------------------------------------------------------
# 1. Stratified, MDS+KNN subset selection
# ---------------------------------------------------------------------------
def select_subset(X, categories, n_total=100, min_per_category=1):
    """
    X: (n_objects, n_dims) SPoSE embedding
    categories: (n_objects,) array-like of category labels
    Returns: array of selected row indices, length n_total

    Within each category, items are projected to 2D via MDS, K-means finds
    `k` centroids in that 2D space, and each centroid is snapped to its
    nearest real item (by 2D coordinates).
    """
    cats = pd.Series(categories)
    cat_counts = cats.value_counts()

    # proportional allocation, but floor every category at min_per_category
    # so rare/atypical categories aren't wiped out by big ones (e.g. animals)
    raw_alloc = (cat_counts / cat_counts.sum() * n_total).round().astype(int)
    alloc = raw_alloc.clip(lower=min_per_category)

    # fix rounding drift so allocations sum exactly to n_total
    diff = n_total - alloc.sum()
    order = alloc.sort_values(ascending=(diff < 0)).index
    i = 0
    while diff != 0:
        c = order[i % len(order)]
        step = 1 if diff > 0 else -1
        if alloc[c] + step >= min_per_category:
            alloc[c] += step
            diff -= step
        i += 1

    selected = []
    for cat, k in alloc.items():
        mask = (cats == cat).values
        idx_in_cat = np.where(mask)[0]
        Xc = X[idx_in_cat]

        k = min(k, len(idx_in_cat))  # can't select more than exist
        if k <= 0:
            continue

        if len(idx_in_cat) <= k:
            selected.extend(idx_in_cat.tolist())
            continue

        chosen_local = _mds_knn_centroids(Xc, k)
        selected.extend(idx_in_cat[chosen_local].tolist())

    return np.array(sorted(set(selected)))[:n_total]


def _mds_knn_centroids(Xc, k):
    """
    Project Xc to 2D via MDS, K-means-cluster the 2D points into k clusters,
    and snap each cluster centroid to its nearest real item (1-NN in 2D).
    Returns local indices (into Xc) of the chosen items, length k
    (topped up by farthest-point fill if centroid collisions reduce the count).
    """
    n_points = Xc.shape[0]
    n_init = "auto"

    mds = MDS(
        n_components=2,
        dissimilarity="euclidean",
        random_state=RNG_SEED,
        n_init=4,
        normalized_stress="auto",
    )
    coords_2d = mds.fit_transform(Xc)

    km = KMeans(n_clusters=k, n_init=n_init, random_state=RNG_SEED).fit(coords_2d)
    nn = NearestNeighbors(n_neighbors=1).fit(coords_2d)
    _, nn_idx = nn.kneighbors(km.cluster_centers_)
    chosen_local = np.unique(nn_idx.flatten())

    # if collisions reduced count below k, top up with farthest-point fill
    if len(chosen_local) < k:
        chosen_local = _farthest_point_fill(coords_2d, chosen_local, k)

    return chosen_local


def _farthest_point_fill(Xc, chosen_local, k):
    """Greedily add the point farthest (min-distance) from the current set."""
    chosen = list(chosen_local)
    remaining = [i for i in range(len(Xc)) if i not in chosen]
    while len(chosen) < k and remaining:
        d = np.linalg.norm(
            Xc[remaining][:, None, :] - Xc[chosen][None, :, :], axis=-1
        ).min(axis=1)
        pick = remaining[int(np.argmax(d))]
        chosen.append(pick)
        remaining.remove(pick)
    return np.array(chosen)


# ---------------------------------------------------------------------------
# 2. Diagnostic: did the subset preserve dimension coverage?
# ---------------------------------------------------------------------------
def dimension_coverage(X, selected):
    full_range = X.max(0) - X.min(0)
    sub_range = X[selected].max(0) - X[selected].min(0)
    coverage = np.divide(sub_range, full_range, out=np.zeros_like(full_range),
                          where=full_range > 0)
    return pd.DataFrame({
        "dim": np.arange(X.shape[1]),
        "full_range": full_range,
        "subset_range": sub_range,
        "coverage_ratio": coverage,
    }).sort_values("coverage_ratio")


# ---------------------------------------------------------------------------
# 3. Diagnostic: RSA check -- does the subset RDM match the full-set submatrix?
# ---------------------------------------------------------------------------
def rdm(X):
    """Pairwise dissimilarity (1 - cosine similarity) matrix."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim = Xn @ Xn.T
    return 1 - sim


def rsa_check(X, selected):
    full_rdm_sub = rdm(X)[np.ix_(selected, selected)]
    direct_rdm = rdm(X[selected])
    iu = np.triu_indices(len(selected), k=1)
    rho, _ = spearmanr(full_rdm_sub[iu], direct_rdm[iu])
    return rho  # should be ~1.0 by construction; sanity check on indexing


# ---------------------------------------------------------------------------
# 4. The real validity test: predictive accuracy on held-out triplets
#    using the SPoSE odd-one-out decision rule (Hebart et al., 2020, eq. 1-2):
#    the predicted odd-one-out is the item NOT in the pair with the
#    highest dot-product similarity among the three pairs.
# ---------------------------------------------------------------------------
def spose_predict_odd_one_out(x_i, x_j, x_k):
    sims = {
        "k": x_i @ x_j,  # pair (i,j) similar -> odd one out is k
        "j": x_i @ x_k,  # pair (i,k) similar -> odd one out is j
        "i": x_j @ x_k,  # pair (j,k) similar -> odd one out is i
    }
    return max(sims, key=sims.get)


def evaluate_on_subset(X, triplets_df, selected):
    """
    triplets_df: columns i, j, k, choice (all as original 0-based indices,
    choice = index of the item chosen as odd one out)
    selected: indices kept in the subset
    """
    sel_set = set(selected.tolist())
    mask = triplets_df[["i", "j", "k"]].isin(sel_set).all(axis=1)
    sub_triplets = triplets_df[mask]

    if len(sub_triplets) == 0:
        return {"n_triplets": 0, "accuracy": None}

    correct = 0
    for _, row in sub_triplets.iterrows():
        i, j, k, choice = int(row["i"]), int(row["j"]), int(row["k"]), int(row["choice"])
        pred_label = spose_predict_odd_one_out(X[i], X[j], X[k])
        pred_idx = {"i": i, "j": j, "k": k}[pred_label]
        correct += int(pred_idx == choice)

    return {
        "n_triplets": len(sub_triplets),
        "accuracy": correct / len(sub_triplets),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR,
                     help="directory containing spose_embedding_49d.txt, "
                          "category_labels.csv, triplet_dataset.csv "
                          "(produced by prepare_things_data.py)")
    ap.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR,
                     help="where to write the selected subset + diagnostics")
    ap.add_argument("--n_total", type=int, default=100)
    ap.add_argument("--min_per_category", type=int, default=1)
    args = ap.parse_args()

    embedding_path = args.data_dir / "spose_embedding_49d.txt"
    categories_path = args.data_dir / "category_labels.csv"
    triplets_path = args.data_dir / "triplet_dataset.csv"
    for p in (embedding_path, categories_path, triplets_path):
        if not p.exists():
            print(f"ERROR: missing {p}. Run prepare_things_data.py first.", file=sys.stderr)
            sys.exit(1)

    X = np.loadtxt(embedding_path)
    cat_df = pd.read_csv(categories_path)
    triplets = pd.read_csv(triplets_path)

    selected = select_subset(X, cat_df["category"].values,
                              n_total=args.n_total,
                              min_per_category=args.min_per_category)
    print(f"Selected {len(selected)} images")

    cov = dimension_coverage(X, selected)
    print("\nWorst-covered dimensions:\n", cov.head(5))

    rho = rsa_check(X, selected)
    print(f"\nRSA sanity check (should be ~1.0): {rho:.4f}")

    result = evaluate_on_subset(X, triplets, selected)
    print(f"\nTriplets fully inside subset: {result['n_triplets']}")
    print(f"SPoSE prediction accuracy on subset triplets: {result['accuracy']}")

    args.results_dir.mkdir(parents=True, exist_ok=True)

    subset_df = cat_df.iloc[selected].copy()
    subset_df.insert(0, "embedding_row", selected)
    subset_df.to_csv(args.results_dir / "selected_subset.csv", index=False)

    cov.to_csv(args.results_dir / "dimension_coverage.csv", index=False)

    pd.DataFrame([{
        "n_selected": len(selected),
        "rsa_sanity_check": rho,
        "n_triplets_in_subset": result["n_triplets"],
        "triplet_accuracy": result["accuracy"],
    }]).to_csv(args.results_dir / "validity_summary.csv", index=False)

    print(f"\nWrote selected_subset.csv, dimension_coverage.csv, "
          f"validity_summary.csv to {args.results_dir}")


if __name__ == "__main__":
    main()
