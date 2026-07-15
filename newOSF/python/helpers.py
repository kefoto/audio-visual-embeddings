"""Python ports of helper_functions/*.m from the Hebart et al. (2020) OSF repo (z2784)."""
import numpy as np
from scipy.spatial.distance import squareform


def squareformq(x):
    """Port of squareformq.m. Matrix<->condensed-vector, ignoring the diagonal.

    MATLAB's column-major lower-triangle fill order is identical to scipy's
    condensed-vector order, so this is a thin wrapper around scipy's squareform.
    """
    x = np.asarray(x)
    if x.ndim == 1 or 1 in x.shape:
        return squareform(np.ravel(x))
    return squareform(x, checks=False)


def embedding2sim(embedding, verbose=True):
    """Port of embedding2sim.m: odd-one-out choice probability between every pair of objects.

    O(n^3); the outer loop stays in Python (n iterations) so the O(n^2) inner
    work per iteration is fully vectorized in numpy. For n=1854 this takes on
    the order of minutes, matching the MATLAB comment ("10-15min on a laptop").
    """
    n = embedding.shape[0]
    sim = embedding @ embedding.T
    esim = np.exp(sim)
    cp = np.zeros((n, n))
    for i in range(n):
        a = esim[i, :]
        denom = a[:, None] + a[None, :] + esim
        ratio = a[:, None] / denom
        ratio[:, i] = 0.0
        np.fill_diagonal(ratio, 0.0)
        cp[i, :] = ratio.sum(axis=1) / (n - 2)
        if verbose and (i + 1) % 200 == 0:
            print(f"    embedding2sim: {i + 1}/{n} rows")
    np.fill_diagonal(cp, 1.0)
    return cp


def embedding2sim_restricted(embedding, obj_idx, pool_idx, verbose=False):
    """Port of the inline similarity computation in make_figures_behavsim.m lines 296-317
    (distinct from embedding2sim.m): odd-one-out probability for pairs within `obj_idx`,
    using only `pool_idx` as the possible distractor set, normalized by len(pool_idx)
    (a fixed constant, NOT adjusted for i/j landing inside the pool - matches the original
    literal `cp = cp/48`, unlike embedding2sim.m's `/(n_objects-2)`).
    """
    sim = embedding @ embedding.T
    esim = np.exp(sim)
    obj_idx = np.asarray(obj_idx)
    pool_idx = np.asarray(pool_idx)
    n_obj, n_pool = len(obj_idx), len(pool_idx)

    cp = np.zeros((n_obj, n_obj))
    for ii in range(n_obj):
        i_global = obj_idx[ii]
        a_ij = esim[i_global, obj_idx]
        a_ik = esim[i_global, pool_idx]
        jk = esim[np.ix_(obj_idx, pool_idx)]
        denom = a_ij[:, None] + a_ik[None, :] + jk
        ratio = a_ij[:, None] / denom
        ratio[:, pool_idx == i_global] = 0.0
        ratio[pool_idx[None, :] == obj_idx[:, None]] = 0.0
        cp[ii, :] = ratio.sum(axis=1) / n_pool
        if verbose and (ii + 1) % 200 == 0:
            print(f"    embedding2sim_restricted: {ii + 1}/{n_obj} rows")
    np.fill_diagonal(cp, 1.0)
    return cp


def fdr_bh(pvals, q=0.05):
    """Port of fdr_bh.m (Groppe), 'pdep' method, matching the .m script's default call.

    Returns (h, crit_p, adj_p) - drops the adj_ci_cvrg output, which the
    MATLAB script never captures via `[~,~,~,p_adj] = fdr_bh(p)`.
    """
    pvals = np.asarray(pvals, dtype=float).ravel()
    m = pvals.size
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    ranks = np.arange(1, m + 1)
    wtd_p = m * p_sorted / ranks
    adj_sorted = np.minimum.accumulate(wtd_p[::-1])[::-1]
    adj_p = np.empty(m)
    adj_p[order] = adj_sorted

    thresh = ranks * q / m
    rej = p_sorted <= thresh
    if rej.any():
        crit_p = p_sorted[rej][-1]
    else:
        crit_p = 0.0
    h = pvals <= crit_p
    return h, crit_p, adj_p


def clustering_algorithm(n_iter, cutoff, sparsedims):
    """Port of clustering_algorithm.m: groups objects by shared top-n_iter dominant
    dimensions for display ordering only (per the original comment, "only for
    visualization purposes" / "somewhat arbitrary way of sorting objects")."""
    order = np.argsort(sparsedims, axis=1)
    top = np.sort(order[:, -n_iter:], axis=1)

    fams = top[:, n_iter - 1].astype(np.int64).copy()
    for i_iter in range(1, n_iter):
        k_iter = n_iter - i_iter
        fams = fams + top[:, i_iter - 1] * (10 ** (2 * k_iter))

    for i_iter in range(1, n_iter + 1):
        ufams, counts = np.unique(fams, return_counts=True)
        mult = 10 ** (2 * i_iter)
        small = ufams[counts < cutoff]
        for fam_val in small:
            mask = fams == fam_val
            fams[mask] = (fams[mask] // mult) * mult + 99 * (10 ** (2 * (i_iter - 1)))

    ufams, counts = np.unique(fams, return_counts=True)
    for fam_val in ufams[counts < cutoff]:
        fams[fams == fam_val] += int(9e6)

    ind = np.argsort(fams, kind="stable")
    return ind, fams[ind]
