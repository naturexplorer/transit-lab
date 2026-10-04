"""Vetting classifiers: the dual-view CNN and the Random Forest baseline.

Both consume the SAME inputs (AstroNet global + local views), so any performance
gap is attributable to model class, not feature engineering. The RF takes the two
views concatenated into one 2202-dim vector.

CNN architecture rationale: one fixed resolution cannot serve both
jobs. Enough resolution to resolve transit shape leaves too short a window to see
context; enough window for context leaves the transit spanning only a few bins.
Two views at two scales, two conv stacks, concatenated before the head.

Data: AstroNet's pre-computed views (scripts/convert_tfrecords.py -> .npz) on
AstroNet's released split. That split is by TCE, so stars leak across splits;
test.npz carries an `unseen_star` mask, and every test metric is reported on both
the full set and the unseen-star subset (see DESIGN.md).
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


TCEs = dict[str, np.ndarray]


""" Section 1: data processing. """
def load_views(path: str | Path) -> TCEs:
    # Load one split written by convert_tfrecords.py fully into memory
    with np.load(path) as f:
        return {key: f[key] for key in f.files}


""" Section 2: scoring and evaluation. """
def vetting_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    # Threshold-free ranking quality: ROC AUC and PR-AUC (average precision)
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "n": len(labels),
        "n_planets": int(labels.sum()),
    }


def evaluate(scores: np.ndarray, tces: TCEs) -> dict[str, dict[str, float]]:
    # Metrics on the whole split, plus on the unseen-star subset when it is flagged.
    results = {"all": vetting_metrics(tces["label"], scores)}
    if "unseen_star" in tces:
        m = tces["unseen_star"]
        results["unseen_star"] = vetting_metrics(tces["label"][m], scores[m])
    return results


""" Section 3: Random Forest. """
def planet_scores(rf: RandomForestClassifier, tces: TCEs) -> np.ndarray:
    # Planet score per TCE: the mean over trees of the planet fraction in its leaf.
    # Can be computes as for one planet, and for many planets.
    if list(rf.classes_) != [0, 1]:
        raise ValueError(f"expected classes [0, 1], got {rf.classes_}")
    proba = rf.predict_proba(rf_features(tces))
    return proba[:, 1]


def rf_features(tces: TCEs) -> np.ndarray:
    # (N, 2202) matrix: each row is one TCE's global view followed by its local view
    return np.concatenate([tces["global_view"], tces["local_view"]], axis=1)


def train_rf(train: TCEs, n_trees: int = 500, seed: int = 0) -> RandomForestClassifier:
    # Fit the RF on the train split. Defaults are sklearn's apart from tree count.
    # Deterministic given `seed`, regardless of n_jobs, which controlls parallel processing.
    rf = RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(rf_features(train), train["label"])
    return rf


""" Section 4: CNN. """
# Best AstroNet configuration: Shallue & Vanderburg (2018), Fig. 7 and Sec. 5.2.
GLOBAL_FILTERS: list[int] = [16, 32, 64, 128, 256]   # number of output channels of each global conv block, in order from bottom to top
LOCAL_FILTERS: list[int] = [16, 32]    # number of output channels of each local conv block, in order from bottom to top
# the number of filters above means that for each filter in that layer, a separate channel will be produced/output from the input channels
CONVS_PER_BLOCK: int = 2    # the number of conv layers in the reusable structure of conv-pool
KERNEL: int = 5             # width of the kernel, ie how many input values (in what locality) to consume for local feature representation
GLOBAL_POOL: int = 5        # max-pool window (how many consecutive values to replace by one), global column
LOCAL_POOL: int = 7         # max-pool window (how many consecutive values to replace by one), local column
POOL_STRIDE: int = 2        # max-pool stride (by how many values to jump for pooling). stride=2 roughly halves the value number
HEAD_LAYERS: int = 4        # number of hidden fully connected layers after the separate convolutions for local and global
HEAD_UNITS: int = 512       # number of neurons in the FC layers
# Training (Sec. 5.2)
BATCH_SIZE, EPOCHS, LR = 64, 300, 1e-5


def conv_block(in_ch: int, out_ch: int, pool_window: int) -> nn.Sequential:
    # CONVS_PER_BLOCK x (Conv1d + ReLU), then one MaxPool1d
    layers: list[nn.Module] = []
    input_ch = in_ch
    for _ in range(CONVS_PER_BLOCK):
        layers.append(nn.Conv1d(input_ch, out_ch, KERNEL, padding="same"))
        layers.append(nn.ReLU())
        input_ch = out_ch
    layers.append(nn.MaxPool1d(pool_window, POOL_STRIDE))
    return nn.Sequential(*layers)


def conv_column(filters: list[int], pool_window: int) -> nn.Sequential:
    # Channels go 1 -> filters[0] -> filters[1] -> ..., ie per block
    blocks, in_ch = [], 1
    for out_ch in filters:
        blocks.append(conv_block(in_ch, out_ch, pool_window))
        in_ch = out_ch
    return nn.Sequential(*blocks)

def _init_like_tf(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)   # PyTorch's name for Glorot-uniform
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class DualViewCNN(nn.Module):
    """Two 1-D conv stacks (global, local) -> concatenate -> dense head -> one logit."""

    def __init__(self) -> None:
        super().__init__()
        self.global_column = conv_column(GLOBAL_FILTERS, GLOBAL_POOL)
        self.local_column = conv_column(LOCAL_FILTERS, LOCAL_POOL)
        n_flat = self._flat_size()
        layers: list[nn.Module] = []
        in_features = n_flat
        for _ in range(HEAD_LAYERS):
            layers.append(nn.Linear(in_features, HEAD_UNITS))
            layers.append(nn.ReLU())
            in_features = HEAD_UNITS
        layers.append(nn.Linear(in_features, 1))
        self.head = nn.Sequential(*layers)
        self.apply(_init_like_tf)

    def _flat_size(self) -> int:
        # Length of the concatenated feature vector, measured with dummy inputs for one example
        with torch.no_grad():
            g = self.global_column(torch.zeros(1, 1, 2001))
            l = self.local_column(torch.zeros(1, 1, 201))
        return g.numel() + l.numel()

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        # (B, 1, 2001), (B, 1, 201) -> logits of shape (B,)
        g = self.global_column(global_view)             # run each view through its column
        l = self.local_column(local_view)
        g_flat = torch.flatten(g, 1)                    # flatten each to (B, C*L)
        l_flat = torch.flatten(l, 1)
        out_cat = torch.cat([g_flat, l_flat], dim=1)    # concatenate along dim 1
        output = self.head(out_cat)                     # apply the head
        return torch.squeeze(output, 1)                 # squeeze to (B,)


def cnn_tensors(tces: TCEs) -> TensorDataset:
    # float32 tensors; unsqueeze(1) adds the channel dimension: (N, 2001) -> (N, 1, 2001)
    g = torch.as_tensor(tces["global_view"], dtype=torch.float32).unsqueeze(1)
    l = torch.as_tensor(tces["local_view"], dtype=torch.float32).unsqueeze(1)
    y = torch.as_tensor(tces["label"], dtype=torch.float32)
    return TensorDataset(g, l, y)


def reflect(g: torch.Tensor, l: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Random horizontal reflection (Sec. 4.2 in paper): each TCE is time-reversed with probability 1/2,
    # both views together, and for each channel identically, since all channels describe same TCE at same index
    mask = torch.rand(len(g)) < 0.5                  # mask of shape (B,)
    mask3d = mask.to(g.device)[:, None, None]        # reshaped to (B, 1, 1)
    # reverse EVERY TCE's (for all channels) time axis to later feed it into torch.where
    g_rev = g.flip(-1)                               
    l_rev = l.flip(-1) 
    # Take the reversed value where flip is True, the original elsewhere.
    # A size-1 dimension in the condition means: the same value for every index along that axis, which is why the mask works
    g_new = torch.where(mask3d, g_rev, g)
    l_new = torch.where(mask3d, l_rev, l)
    return g_new, l_new


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def cnn_scores(model: DualViewCNN, tces: TCEs, batch_size: int = 512) -> np.ndarray:
    # Planet score per TCE: sigmoid of the logit. Same role as planet_scores for the RF.
    model.eval()
    device = next(model.parameters()).device
    out = []
    for g, l, _ in DataLoader(cnn_tensors(tces), batch_size=batch_size):
        g, l = g.to(device), l.to(device)
        logits = model(g, l)
        probs = torch.sigmoid(logits)
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


def train_cnn(train: TCEs, val: TCEs | None = None, seed: int = 0,
              epochs: int = EPOCHS) -> Tuple[DualViewCNN, int]:
    # Fixed epoch count: no decision is taken on val, which is logged for monitoring only.
    device = pick_device()
    torch.manual_seed(seed)                 # sets global seed; makes weight init and reflections reproducible
    model = DualViewCNN().to(device)
    loader = DataLoader(cnn_tensors(train), batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed)) 
    # DataLoader is an iterable wrapping a dataset; it gives one batch per iteration
    # shuffling mixes the N inputs and forms batches of 64 using generator
    # it collates the 64 examples, stacking each component along a new dimension 0:
    #  global views: (64, 1, 2001), local views: (64, 1, 201), labels: (64,)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    best_auc, best_state, best_epoch, patience, stale, = -1.0, None, 0, 15, 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for g, l, y in loader:     # unpack the batch triple as described above
            g, l, y = g.to(device), l.to(device), y.to(device)  # copy the batch from CPU memory, where the dataset lives, to the device holding the model
            g, l = reflect(g, l)    # add variety to the training data; dataset doesn't grow though.
            # Forward pass
            logits = model(g, l)
            # Compute prediction error for the whole batch (mean)
            loss = loss_fn(logits, y)
            # Backpropagation
            opt.zero_grad()     # reset the parameter gradients to 0 to avoid accumulation (we so step every batch)
            loss.backward()     # perform backpropagation and write the gradients into parameters
            opt.step()          # perform gradient descent using the backprop's results
            total += loss.item() * len(y) # turn the batch mean loss back into the batch summed loss

        print(f"epoch: {epoch + 1:3d}  train loss: {total / len(train['label']):.4f}", end="") # average loss per example;
        # * len(y) is done above instead of averaging the batch mean losses because the last batch might be smaller than 64 (drop_last=False)
        if val is not None:
            metrics = vetting_metrics(val['label'], cnn_scores(model, val))
            auc = metrics['auc']
            msg = f"  val auc: {auc:.4f}"
            msg += f"  val pr_auc: {metrics['pr_auc']:.4f}"
            print(msg)
            # early stopping logic
            if auc > best_auc:
                best_auc, best_epoch, stale = auc, epoch + 1, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break

    print(f"epoch for best auc: {best_epoch}")
    if best_state is not None:
       model.load_state_dict(best_state)
    return (model, best_epoch)