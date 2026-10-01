"""Here I convert AstroNet's pre-computed TFRecords into one .npz file per split.
It's run once, from the repo root:  python scripts/convert_tfrecords.py
Why: it keeps TFRecord parsing out of transit_lab's
runtime dependencies. Training code only ever reads the .npz files written here.
So I am commiting the archives, so that results can be reproduced.
"""

import glob
from itertools import combinations
from pathlib import Path
from typing import cast
 
import numpy as np
from tfrecord.reader import tfrecord_loader
 
DATA_DIR = Path("astronet_data")
GLOBAL_LEN, LOCAL_LEN = 2001, 201
 
# Binary vetting target: planet candidate vs everything else. UNK must not appear
# in the released sets; any other label is a hard error, never silently dropped.
LABELS = {"PC": 1, "AFP": 0, "NTP": 0}
 
# Every TFRecord feature is a list; these name the element type, not the length.
DESCRIPTION = {
    "global_view": "float",
    "local_view": "float",
    "av_training_set": "byte",
    "kepid": "int",
    "tce_plnt_num": "int",
}
SPLITS = {"train": "train-*", "val": "val-*", "test": "test-*"}
 

def convert_split(pattern: str) -> dict[str, np.ndarray]:
    """Read every record in the files matching `pattern` into stacked arrays."""
    files = sorted(glob.glob(str(DATA_DIR / pattern)))
    if not files:
        raise FileNotFoundError(f"no files match {DATA_DIR / pattern}")


    # Accumulate in lists and stack once: appending to an ndarray copies it each time.
    global_views: list[np.ndarray] = []
    local_views: list[np.ndarray] = []
    labels: list[int] = []
    kepids: list[int] = []
    plnt_nums: list[int] = []

    for file in files:
        for raw in tfrecord_loader(file, None, DESCRIPTION):
            # The library's return annotation is wrong (union outside Iterable);
            # without sequence_description every item is a plain dict.
            ex = cast(dict[str, np.ndarray], raw)

            kepid = int(ex["kepid"][0])  # because scalars are stored as length-1 lists
            label_str = bytes(ex["av_training_set"]).decode()
            if label_str not in LABELS:
                raise ValueError(f"{file}: kepid {kepid} has label {label_str!r}")

            # The file format doesn't fix list lengths; AstroNet's convention does.
            if ex["global_view"].shape != (GLOBAL_LEN,):
                raise ValueError(f"{file}: kepid {kepid} global_view {ex['global_view'].shape}")
            if ex["local_view"].shape != (LOCAL_LEN,):
                raise ValueError(f"{file}: kepid {kepid} local_view {ex['local_view'].shape}")

            global_views.append(ex["global_view"])
            local_views.append(ex["local_view"])
            labels.append(LABELS[label_str])
            kepids.append(kepid)
            plnt_nums.append(int(ex["tce_plnt_num"][0]))

    # One TCE = one (star, signal index) pair; a repeat is a data error, not a choice.
    n_unique = len(set(zip(kepids, plnt_nums)))
    if n_unique != len(kepids):
        raise ValueError(f"{pattern}: {len(kepids) - n_unique} duplicate TCE records")

    return {
        # produce one contiguous block of N×2001 floats instead of list of pointers, plus metadata: shape (N, 2001)
        "global_view": np.stack(global_views).astype(np.float32),
        "local_view": np.stack(local_views).astype(np.float32),
        "label": np.array(labels, dtype=np.int8),
        "kepid": np.array(kepids, dtype=np.int64),
        "tce_plnt_num": np.array(plnt_nums, dtype=np.int64),
    }


def main() -> None:
    splits = {name: convert_split(pattern) for name, pattern in SPLITS.items()}
 
    # AstroNet split by TCE, not by star, so stars leak across splits. Rather than
    # re-split (losing comparability with the paper), I will flag the star-disjoint
    # subsets so evaluation can report AUC on both and measure the leak.
    #   test: star absent from train AND val. I will compare it with mixed testing
    train_stars = np.unique(splits["train"]["kepid"])
    train_val_stars = np.union1d(train_stars, splits["val"]["kepid"])
    splits["test"]["star_disjoint"] = ~np.isin(splits["test"]["kepid"], train_val_stars)

    # Leakage check: does any star appear in more than one split?
    # we don't want two of its transiting planets to land in train and test
    print("\nstars shared between splits:") 
    for a, b in combinations(splits, 2):
        shared = set(splits[a]["kepid"].tolist()) & set(splits[b]["kepid"].tolist())
        print(f"  {a} & {b}: {len(shared)}")
 
    for name, data in splits.items():
        # keys never include allow_pickle
        np.savez_compressed(DATA_DIR / f"{name}.npz", **data)  # pyright: ignore[reportArgumentType]
 
        labels, kepids = data["label"], data["kepid"]
        line = (f"{name:5s}  TCEs ={len(labels):6d}  planets ={int(labels.sum()):5d} "
                f"({labels.mean():.1%})  stars ={np.unique(kepids).size:5d}")
        if "star_disjoint" in data:
            mask = data["star_disjoint"]
            line += (f"  | star-disjoint: TCEs ={int(mask.sum()):5d}  "
                     f"planets ={int(labels[mask].sum()):4d}  stars ={np.unique(kepids[mask]).size:5d}")
            # say  labels = np.array([1, 0, 0, 1, 0])
            # and  mask   = np.array([True, False, True, True, False])
            # then labels[mask] gives array([1, 0, 1])
        print(line)

if __name__ == "__main__":
    main()