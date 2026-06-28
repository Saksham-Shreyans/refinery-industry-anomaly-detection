"""
tep_full_pipeline_v5.py  —  TEP Multi-Class Fault Classifier
=============================================================
Stage 2 upgrade: binary detection  →  21-class identification.

KEY CHANGES vs v4
─────────────────
 • FaultClassifierLSTM  : output head changed from  Linear→1  to  Linear→21
 • Labels               : integer fault number 0-20  (not binary 0/1)
 • Loss                 : CrossEntropyLoss           (not BCEWithLogitsLoss)
 • WindowDataset        : y is LongTensor of shape (N,)
 • evaluate()           : per-fault FDR (precision) table + confusion summary
 • Model files          : tep_model_v5.pth  /  tep_norm_v5.npz
                          (saved separately so v4 binary model is untouched)

WHAT IS FDR HERE?
─────────────────
 FDR (Fault Detection Rate) = precision per fault class
    = TP_i / (TP_i + FP_i)   for fault i
 i.e. of all windows the model labels as fault-i, what fraction truly are fault-i.
 We also report Recall_i = TP_i / (total true fault-i windows)
 and the confusion matrix diagonal.

HOW TO RUN
──────────
 Train from scratch:
     set  LOAD_PRETRAINED = False   (default below)
     python tep_full_pipeline_v5.py

 Evaluate only (needs tep_model_v5.pth + tep_norm_v5.npz):
     set  LOAD_PRETRAINED = True
     python tep_full_pipeline_v5.py

 Required files (same folder):
     fault_free_training.RData
     faulty_training.RData
     faulty_testing.RData
     (TEP_* capitalized names also work — rename first)
"""

import os, random
import numpy as np
import pandas as pd
import pyreadr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings("ignore")

# ── REPRODUCIBILITY ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ── HYPER-PARAMS ─────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 50       # time-steps fed to LSTM per window
BATCH_SIZE      = 128      # larger batch → more stable CE gradients
EPOCHS          = 200      # CE converges faster than BCE; 200 is plenty
LR              = 0.001
HIDDEN_SIZE     = 128      # wider hidden layer for 21-way classification
NUM_CLASSES     = 21       # faults 0-20  (0 = normal)
TRAIN_RUNS      = 450      # simulation runs used for training
VAL_RUNS        = 50       # held-out runs for early stopping

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "tep_model_v5.pth")
NORM_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "tep_norm_v5.npz")

# Faults to SKIP in training/eval (literature: near-impossible to detect)
SKIP_FAULTS = {3, 9, 15}

# ── DATA LOADING ─────────────────────────────────────────────────────────────
def load(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing: {filename}")
    print(f"  Loading {filename} ...", end=" ", flush=True)
    r  = pyreadr.read_r(filename)
    df = r[list(r.keys())[0]]
    print(f"OK  ({len(df):,} rows)")
    return df

def sensor_cols(df):
    return [c for c in df.columns
            if c.startswith("xmeas_") or c.startswith("xmv_")]

# ── NORMALISATION ────────────────────────────────────────────────────────────
def fit_normalisation(df_normal):
    """Fit mean/std from fault-free training data only."""
    cols  = sensor_cols(df_normal)
    means = df_normal[cols].mean().values.astype(np.float32)
    stds  = df_normal[cols].std().values.astype(np.float32)
    stds[stds < 1e-6] = 1.0
    return means, stds

# ── WINDOW EXTRACTION ────────────────────────────────────────────────────────
def extract_windows(df, means, stds, start_run=1, max_runs=20, stride=5,
                    fault_intro_train=20, fault_intro_test=160,
                    skip_faults=SKIP_FAULTS):
    """
    For each (fault, run) pair in [start_run, start_run+max_runs),
    slide a window of SEQUENCE_LENGTH over the post-normalised series.
    The window label = fault number (integer 0-20).

    start_run : first simulation run to include (1-indexed, inclusive)
    max_runs  : how many runs to include from start_run

    For FAULTY runs we skip samples before the fault intro so the model
    only sees confirmed-fault windows (matches Medium article approach).
    For fault 0 (normal) we use ALL samples.
    """
    cols = sensor_cols(df)
    wins = []

    faults   = sorted(int(f) for f in df["faultNumber"].unique()
                      if int(f) not in skip_faults)
    all_runs = sorted(int(r) for r in df["simulationRun"].unique())
    # Select the slice [start_run-1 : start_run-1+max_runs]
    runs = all_runs[start_run - 1 : start_run - 1 + max_runs]

    # Detect fault intro: training data has 500 samples/run, test has 960
    sample_len = len(df[df["simulationRun"] == df["simulationRun"].iloc[0]])
    fault_intro = fault_intro_train if sample_len <= 500 else fault_intro_test

    for fault in faults:
        for run in runs:
            run_df = (df[(df["faultNumber"] == fault) &
                         (df["simulationRun"] == run)]
                      .sort_values("sample"))
            if run_df.empty:
                continue

            data  = run_df[cols].values.astype(np.float32)
            data  = (data - means) / stds
            total = len(data)

            start_at = fault_intro if fault != 0 else 0

            for end in range(max(SEQUENCE_LENGTH, start_at + 1), total + 1, stride):
                window = data[end - SEQUENCE_LENGTH: end]
                wins.append((window, fault))

    return wins

# ── DATASET ───────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    def __init__(self, windows):
        self.X = torch.tensor(
            np.stack([w for w, _ in windows]), dtype=torch.float32)
        # CrossEntropyLoss expects Long integer class indices, shape (N,)
        self.y = torch.tensor([l for _, l in windows], dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── MODEL ─────────────────────────────────────────────────────────────────────
class FaultClassifierLSTM(nn.Module):
    """
    Two-layer LSTM → 21-way softmax classifier.

    v4 had:  LSTM → Linear(hidden,32) → ReLU → Dropout → Linear(32,1)
    v5 has:  LSTM → Linear(hidden,64) → ReLU → Dropout → Linear(64,21)
                                                           ↑
                                              21 classes, not 1 logit
    """
    def __init__(self, n_sensors, hidden=128, n_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors, hidden_size=hidden,
            num_layers=2, batch_first=True, dropout=0.3
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)   # raw logits; softmax applied by loss
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # shape: (batch, 21)

    def predict_class(self, x):
        """Return predicted fault number (argmax of softmax)."""
        with torch.no_grad():
            logits = self.forward(x)
            return int(torch.argmax(logits, dim=-1).item())

    def predict_proba(self, x):
        """Return softmax probabilities for all 21 classes."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

# ── TRAINING ─────────────────────────────────────────────────────────────────
def train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    """
    CrossEntropyLoss — takes raw logits (shape B×21) and integer targets (B,).
    No pos_weight needed since we balanced the dataset by skipping pre-fault
    normal samples inside faulty runs.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5)

    best_val  = float("inf")
    patience  = 10
    no_improv = 0

    for epoch in range(1, epochs + 1):
        # ── train pass ──
        model.train()
        train_loss = 0.0
        correct = total = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)              # (B, 21)
            loss   = criterion(logits, y_batch)  # CE: logits vs class indices
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            preds   = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total   += len(y_batch)

        train_loss /= len(train_loader)
        train_acc   = correct / total * 100

        # ── val pass ──
        model.eval()
        val_loss = 0.0
        vcorrect = vtotal = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits    = model(X_batch)
                val_loss += criterion(logits, y_batch).item()
                preds     = logits.argmax(dim=1)
                vcorrect += (preds == y_batch).sum().item()
                vtotal   += len(y_batch)

        val_loss /= len(val_loader)
        val_acc   = vcorrect / vtotal * 100
        scheduler.step(val_loss)

        print(f"  Epoch {epoch:03d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.1f}%")

        if val_loss < best_val - 1e-4:
            best_val  = val_loss
            no_improv = 0
            torch.save(model.state_dict(), MODEL_FILE)
            print(f"    → Best val_loss={best_val:.4f}  val_acc={val_acc:.1f}%  Saved.")
        else:
            no_improv += 1
            if no_improv >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(MODEL_FILE))
    return model

# ── EVALUATION ───────────────────────────────────────────────────────────────
def evaluate(model, df_test_faulty, means, stds):
    """
    Per-fault FDR (precision) and recall table.
    Uses ALL test runs (not just run 1) for a robust estimate.
    FDR_i = TP_i / (TP_i + FP_i)   ← precision: of windows called fault-i,
                                       how many really are fault-i?
    Recall_i = TP_i / total_true_i  ← of all true fault-i windows,
                                       how many did we catch?
    """
    model.eval()
    cols     = sensor_cols(df_test_faulty)
    n_s      = len(cols)
    faults   = sorted(int(f) for f in df_test_faulty["faultNumber"].unique()
                      if int(f) not in SKIP_FAULTS)
    runs     = sorted(int(r) for r in df_test_faulty["simulationRun"].unique())

    # confusion matrix  [true_class, pred_class]
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    fault_intro = 160   # test data: fault introduced at sample 160

    print(f"\n  Running inference over {len(faults)} faults × "
          f"{len(runs)} runs …")

    with torch.no_grad():
        for fault in faults:
            for run in runs:
                run_df = (df_test_faulty[
                    (df_test_faulty["faultNumber"] == fault) &
                    (df_test_faulty["simulationRun"] == run)]
                    .sort_values("sample"))
                if run_df.empty:
                    continue

                data  = run_df[cols].values.astype(np.float32)
                data  = (data - means) / stds
                total = len(data)

                # Only evaluate on POST-fault window (samples ≥ fault_intro)
                # for faulty runs, or ALL samples for fault 0
                start_at = fault_intro if fault != 0 else 0

                for end in range(max(SEQUENCE_LENGTH, start_at + 1),
                                 total + 1, 5):          # stride=5 for speed
                    window = data[end - SEQUENCE_LENGTH: end]
                    X      = torch.tensor(window).unsqueeze(0)
                    logits = model(X)                     # (1, 21)
                    pred   = int(logits.argmax(dim=1).item())
                    true_c = fault
                    conf[true_c, pred] += 1

    # ── Per-fault metrics ────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  {'Fault':<8} {'Description':<38} {'FDR(prec)':>10} "
          f"{'Recall':>8} {'Count':>7}")
    print("  " + "-"*68)

    fdr_list = []
    for f in faults:
        true_count = conf[f, :].sum()          # total true windows for fault f
        pred_count = conf[:, f].sum()          # total windows predicted as f
        tp         = conf[f, f]                # correctly classified as f

        fdr    = tp / pred_count * 100 if pred_count > 0 else 0.0
        recall = tp / true_count * 100 if true_count > 0 else 0.0
        fdr_list.append(fdr)

        desc = {
            0:"Normal",1:"A/C feed ratio step",2:"B composition step",
            4:"Reactor coolant step",5:"Condenser coolant step",
            6:"A feed loss step",7:"C header pressure step",
            8:"A B C composition random",10:"C feed temp random",
            11:"Reactor coolant random",12:"Condenser coolant random",
            13:"Reaction kinetics drift",14:"Reactor cooling valve stuck",
            16:"Unknown",17:"Unknown",18:"Unknown",19:"Unknown",20:"Unknown"
        }.get(f, f"Fault {f}")

        bar = "█" * int(fdr / 5) + "░" * (20 - int(fdr / 5))
        flag = "  ⚠ hard" if f in {3,9,15} else ""
        print(f"  F{str(f).zfill(2)}     {desc:<38} {fdr:>9.1f}%"
              f" {recall:>7.1f}% {true_count:>7,}{flag}")

    avg_fdr = np.mean(fdr_list)
    print("  " + "-"*68)
    print(f"  {'AVERAGE FDR':<48} {avg_fdr:>9.1f}%")
    print("="*72)

    # ── Compact confusion matrix (predicted vs true) ──────────────────────
    print("\n  CONFUSION MATRIX  (rows=true fault, cols=predicted fault)")
    print("  Showing faults:", faults)
    header = "       " + "".join(f"{f:>5}" for f in faults)
    print("  " + header)
    for r in faults:
        row_str = f"  F{str(r).zfill(2)}  " + "".join(
            f"{conf[r,c]:>5}" for c in faults)
        print(row_str)

    return conf, fdr_list

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    LOAD_PRETRAINED = True  # ← set True to skip training and just evaluate

    print("=" * 65)
    print("  TEP Fault Classifier v5 — Multi-Class (21 outputs)")
    print("=" * 65)

    print("\nLoading data …")
    df_test_faulty = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "faulty_testing.RData"))
    cols      = sensor_cols(df_test_faulty)
    n_sensors = len(cols)

    model = FaultClassifierLSTM(n_sensors=n_sensors, hidden=HIDDEN_SIZE,
                                n_classes=NUM_CLASSES)

    if not LOAD_PRETRAINED:
        df_normal = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "fault_free_training.RData"))
        df_faulty = load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "faulty_training.RData"))

        print("\nFitting normalisation on fault-free training data …")
        means, stds = fit_normalisation(df_normal)
        np.savez(NORM_FILE, means=means, stds=stds)
        print(f"  Saved {NORM_FILE}")

        print(f"\nExtracting TRAINING windows  "
              f"(runs 1-{TRAIN_RUNS}, stride=5) …")
        wins_normal = extract_windows(
            df_normal, means, stds,
            start_run=1, max_runs=TRAIN_RUNS, stride=5)
        wins_faulty = extract_windows(
            df_faulty, means, stds,
            start_run=1, max_runs=TRAIN_RUNS, stride=5)
        all_wins = wins_normal + wins_faulty

        print(f"\nExtracting VALIDATION windows  "
              f"(runs {TRAIN_RUNS+1}-{TRAIN_RUNS+VAL_RUNS}) …")
        val_normal = extract_windows(
            df_normal, means, stds,
            start_run=TRAIN_RUNS+1, max_runs=VAL_RUNS, stride=5)
        val_faulty = extract_windows(
            df_faulty, means, stds,
            start_run=TRAIN_RUNS+1, max_runs=VAL_RUNS, stride=5)
        val_wins = val_normal + val_faulty

        # Class counts
        from collections import Counter
        counts = Counter(l for _, l in all_wins)
        print("\n  Class distribution in training windows:")
        for cl in sorted(counts):
            print(f"    Fault {cl:02d}: {counts[cl]:,}")

        random.shuffle(all_wins)
        random.shuffle(val_wins)

        train_ds = WindowDataset(all_wins)
        val_ds   = WindowDataset(val_wins)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)

        print(f"\n  Training  : {len(train_ds):,} windows")
        print(f"  Validation: {len(val_ds):,} windows")
        print(f"  batch_size={BATCH_SIZE}   epochs={EPOCHS}\n")

        model = train(model, train_loader, val_loader,
                      epochs=EPOCHS, lr=LR)

        del df_normal, df_faulty

    else:
        norm  = np.load(NORM_FILE)
        means = norm["means"]
        stds  = norm["stds"]
        model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
        print(f"  Loaded weights from {MODEL_FILE}")

    print("\n" + "=" * 65)
    print("  EVALUATION — per-fault FDR on unseen testing data")
    print("=" * 65 + "\n")

    evaluate(model, df_test_faulty, means, stds)

if __name__ == "__main__":
    main()
