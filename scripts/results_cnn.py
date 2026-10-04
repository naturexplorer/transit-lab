"""Evaluate the performance of a saved CNN from a .pt file. Report vetting metrics. """

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transit_lab.vetting import DualViewCNN, evaluate, load_views, pick_device, cnn_scores

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "astronet_data"

def load_cnn(path: str | Path) -> DualViewCNN:
    device = pick_device()
    checkpoint = torch.load(path, map_location=device)   # map_location: load straight onto this machine's device
    model = DualViewCNN().to(device)                     # rebuild the architecture from code...
    model.load_state_dict(checkpoint["state_dict"])      # put the saved weights into it
    model.eval()                                         # inference mode, before scoring
    return model


def test_cnn(path: str | Path, split: str = "test"):
    # the eval can either be test or val TCEs
    cnn = load_cnn(path)
    views = load_views(DATA_DIR / f"{split}.npz")
    for subset, m in evaluate(cnn_scores(cnn, views), views).items():
        print(f"    {split}/{subset:11s}  AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}")


def main() -> None:
    test_cnn(ROOT / "models" / "cnn_seed0.pt", split="test") # can also do on 'val'


if __name__ == "__main__":
    main()