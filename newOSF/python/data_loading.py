"""Python port of the Setup + Load-relevant-data section (lines 1-91) of make_figures_behavsim.m.

All indices are converted from MATLAB's 1-based convention to Python's 0-based
convention at load time, so everything downstream is plain 0-indexed numpy.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io as sio

from matutils import cellnum, cellstr, to_1d

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
VARIABLE_DIR = BASE_DIR / "variables"

# magic 1-indexed color picks from make_figures_behavsim.m lines 69-70, kept verbatim
_COLOR_PICK_1INDEXED = [
    1, 20, 3, 38, 9, 7, 62, 57, 13, 6, 24, 25, 50, 48, 36, 53, 46, 28, 62, 18,
    15, 58, 2, 11, 40, 45, 27, 55, 36, 30, 34, 31, 41, 16, 27, 61, 17, 36, 57, 25, 63,
]


def _load_colors():
    lines = (VARIABLE_DIR / "colors.txt").read_text().splitlines()
    col = np.array([[int(h[1:][i:i + 2], 16) / 255 for i in (0, 2, 4)] for h in lines if h.strip()])
    col = col[1:]  # drop black (matches MATLAB col(1,:) = [])
    col[[0, 1, 2]] = col[[1, 2, 0]]  # rotate first three rows

    pick = np.array(_COLOR_PICK_1INDEXED) - 1
    colors = col[pick]
    n_rest = 49 - len(colors)
    rest_idx = np.arange(7, 7 + n_rest)  # MATLAB 8:(56-len(colors)), 1-indexed -> 0-indexed
    colors = np.vstack([colors, col[rest_idx]])
    colors[45] = colors[45] - 0.2  # dimension 46 (1-indexed) is too bright
    return colors


def _align_images(im_raw, imwords_raw, unique_id_list):
    imwords_list = cellstr(imwords_raw)
    pos = {w: i for i, w in enumerate(imwords_list)}
    order = [pos[w] for w in unique_id_list]
    im_flat = to_1d(im_raw)
    return [im_flat[k] for k in order]


def load_all():
    ns = SimpleNamespace()

    ns.spose_embedding49 = np.loadtxt(DATA_DIR / "spose_embedding_49d_sorted.txt")
    ns.dot_product49 = ns.spose_embedding49 @ ns.spose_embedding49.T

    ns.spose_sim = sio.loadmat(DATA_DIR / "spose_similarity.mat")["spose_sim"]
    ns.dissim = 1 - ns.spose_sim

    raw_test = np.loadtxt(DATA_DIR / "data1854_batch5_test10.txt").astype(int)  # already 0-indexed

    rdm = sio.loadmat(DATA_DIR / "RDM48_triplet.mat")
    ns.RDM48_triplet = rdm["RDM48_triplet"]
    rdm_sh = sio.loadmat(DATA_DIR / "RDM48_triplet_splithalf.mat")
    ns.RDM48_triplet_split1 = rdm_sh["RDM48_triplet_split1"]
    ns.RDM48_triplet_split2 = rdm_sh["RDM48_triplet_split2"]

    ns.typicality_data27 = sio.loadmat(DATA_DIR / "typicality_data27.mat")
    # kept as float (NaN preserved) - 10 of 27 categories have no matching dimension;
    # category27_subind selects exactly the 17 that do, cast to int after that selection
    ns.best_match27 = to_1d(ns.typicality_data27["best_match27"]) - 1
    ns.categories27 = cellstr(ns.typicality_data27["categories27"])
    ns.category27_ind = [ind.astype(int) - 1 for ind in cellnum(ns.typicality_data27["category27_ind"])]
    ns.category27_subind = to_1d(ns.typicality_data27["category27_subind"]).astype(int) - 1
    ns.category27_typicality_rating_normed = cellnum(ns.typicality_data27["category27_typicality_rating_normed"])

    ns.ratings_translated_all = sio.loadmat(DATA_DIR / "dimension_ratings.mat")["ratings_translated_all"]

    dimlabel_raw = sio.loadmat(DATA_DIR / "dimlabel_answers.mat")["dimlabel_answers"]
    ns.dimlabel_answers = np.array(
        [[str(dimlabel_raw[i, j][0]) for j in range(dimlabel_raw.shape[1])] for i in range(dimlabel_raw.shape[0])]
    )

    ns.category_mat_manual = sio.loadmat(DATA_DIR / "category_mat_manual.mat")["category_mat_manual"]
    ns.sensevec_augmented = sio.loadmat(DATA_DIR / "sensevec_augmented_with_wordvec.mat")["sensevec_augmented"]

    # resort raw triplet indices: sortind[i_obj] (1-indexed, MATLAB) is the OLD raw label
    # that should become NEW label i_obj. Build the inverse map and apply it.
    sortind = to_1d(sio.loadmat(VARIABLE_DIR / "sortind.mat")["sortind"]).astype(int) - 1
    remap = np.empty(1854, dtype=int)
    remap[sortind] = np.arange(1854)
    ns.triplet_testdata49 = remap[raw_test]

    ns.labels = cellstr(sio.loadmat(VARIABLE_DIR / "labels.mat")["labels"])
    ns.labels_short = cellstr(sio.loadmat(VARIABLE_DIR / "labels_short.mat")["labels_short"])
    ns.colors = _load_colors()

    ns.words = cellstr(sio.loadmat(VARIABLE_DIR / "words.mat")["words"])
    ns.unique_id = cellstr(sio.loadmat(VARIABLE_DIR / "unique_id.mat")["unique_id"])
    ns.words48 = cellstr(sio.loadmat(VARIABLE_DIR / "words48.mat")["words48"])

    im_mat = sio.loadmat(VARIABLE_DIR / "im.mat")
    ns.im = _align_images(im_mat["im"], im_mat["imwords"], ns.unique_id)

    words_pos = {w: i for i, w in enumerate(ns.words)}
    ns.wordposition48 = [words_pos[w] for w in ns.words48]

    return ns


if __name__ == "__main__":
    data = load_all()
    for k, v in vars(data).items():
        shape = getattr(v, "shape", None)
        length = len(v) if hasattr(v, "__len__") else None
        print(f"{k}: shape={shape} len={length} type={type(v).__name__}")
