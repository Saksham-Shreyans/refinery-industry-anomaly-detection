"""
evaluate_mixed_dataset.py  —  TEP Fault Detection on Mixed-Timing Test Data
=============================================================================
Evaluates the trained LSTM model against the mixed dataset produced by
create_mixed_test_dataset.py.

KEY DIFFERENCE FROM ORIGINAL EVALUATION
----------------------------------------
The original evaluate() in tep_full_pipeline_v5.py hardcodes fault_intro=160.
Here we read the TRUE fault_intro_sample from the metadata CSV (one value per
run), so windows are labelled correctly regardless of WHEN the fault starts.

A window is labelled:
    NORMAL (class 0)  → window ends BEFORE fault_intro_sample
    FAULT  (class f)  → window ends AT or AFTER fault_intro_sample

USAGE
-----
    python evaluate_mixed_dataset.py

REQUIRES (same folder)
    tep_model_v5.pth              ← trained weights
    tep_norm_v5.npz               ← normalisation params
    mixed_test_dataset.csv        ← built by create_mixed_test_dataset.py
    mixed_test_metadata.csv       ← built by create_mixed_test_dataset.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "..", "data", "processed", "mixed_test_dataset.csv")
META_FILE  = os.path.join(BASE_DIR, "..", "data", "processed", "mixed_test_metadata.csv")
MODEL_FILE = os.path.join(BASE_DIR, "..", "models", "tep_model_v5.pth")
NORM_FILE  = os.path.join(BASE_DIR, "..", "models", "tep_norm_v5.npz")

# ─── MODEL PARAMS (must match training) ──────────────────────────────────────
SEQUENCE_LENGTH = 50
HIDDEN_SIZE     = 128
NUM_CLASSES     = 21
SKIP_FAULTS     = {3, 9, 15}
STRIDE          = 5


# ─── MODEL DEFINITION (must match tep_full_pipeline_v5.py exactly) ───────────
# Architecture: LSTM → Linear(hidden,64) → ReLU → Dropout → Linear(64,n_classes)
# Saved keys:   head.0.weight / head.0.bias / head.3.weight / head.3.bias
class FaultClassifierLSTM(nn.Module):
    def __init__(self, n_sensors, hidden=128, n_classes=21):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors, hidden_size=hidden,
            num_layers=2, batch_first=True, dropout=0.3
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)   # raw logits
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def sensor_cols(df):
    return [c for c in df.columns
            if c.startswith("xmeas_") or c.startswith("xmv_")]


def load_model_and_norm():
    for f in [MODEL_FILE, NORM_FILE]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing: {f}  — run training first.")
    norm  = np.load(NORM_FILE)
    means = norm["means"]
    stds  = norm["stds"]
    return means, stds


# ─── EVALUATION ──────────────────────────────────────────────────────────────
def evaluate_mixed(model, df_data, df_meta, means, stds):
    """
    Per-fault detection metrics on the mixed-timing dataset.

    Window labelling uses per-run fault_intro_sample (not hardcoded 160):
      • window ends at sample s:
          s <  fault_intro  →  true label = 0  (normal)
          s >= fault_intro  →  true label = fault_number
    """
    model.eval()
    scols  = sensor_cols(df_data)
    faults = sorted(int(f) for f in df_data["faultNumber"].unique()
                    if int(f) not in SKIP_FAULTS)

    # Build per-run lookup: run_id → fault_intro_sample
    intro_map = dict(zip(df_meta["simulationRun"],
                         df_meta["fault_intro_sample"]))
    fault_map = dict(zip(df_meta["simulationRun"],
                         df_meta["faultNumber"]))

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # Per-intro-time breakdown
    intro_times = sorted(df_meta["fault_intro_sample"].unique())
    per_intro   = {t: {"tp": 0, "total": 0} for t in intro_times}

    all_runs = sorted(df_data["simulationRun"].unique())
    print(f"\n  Running inference over {len(faults)} faults × "
          f"{len(all_runs)} runs …")

    with torch.no_grad():
        for run_id in all_runs:
            run_df = (df_data[df_data["simulationRun"] == run_id]
                      .sort_values("sample"))
            if run_df.empty:
                continue

            fault_intro = intro_map[run_id]
            true_fault  = fault_map[run_id]

            data  = run_df[scols].values.astype(np.float32)
            data  = (data - means) / stds
            total = len(data)

            for end in range(SEQUENCE_LENGTH, total + 1, STRIDE):
                window    = data[end - SEQUENCE_LENGTH: end]
                X         = torch.tensor(window).unsqueeze(0)
                logits    = model(X)
                pred      = int(logits.argmax(dim=1).item())
                end_sample = end   # 1-indexed equivalent (window covers 1..end)

                # True label based on whether window end is post-fault
                true_c = true_fault if end_sample >= fault_intro else 0

                conf[true_c, pred] += 1

                # Track per-intro-time TP for faulty windows
                if true_c == true_fault and true_c != 0:
                    per_intro[fault_intro]["total"] += 1
                    if pred == true_c:
                        per_intro[fault_intro]["tp"] += 1

    # ── Per-fault metrics ────────────────────────────────────────────────────
    fault_desc = {
        0:"Normal",1:"A/C feed ratio step",2:"B composition step",
        4:"Reactor coolant step",5:"Condenser coolant step",
        6:"A feed loss step",7:"C header pressure step",
        8:"A B C composition random",10:"C feed temp random",
        11:"Reactor coolant random",12:"Condenser coolant random",
        13:"Reaction kinetics drift",14:"Reactor cooling valve stuck",
        16:"Unknown",17:"Unknown",18:"Unknown",19:"Unknown",20:"Unknown"
    }

    print("\n" + "=" * 72)
    print(f"  {'Fault':<8} {'Description':<38} {'FDR(prec)':>10} "
          f"{'Recall':>8} {'Windows':>8}")
    print("  " + "-" * 68)

    fdr_list = []
    for f in faults:
        true_count = conf[f, :].sum()
        pred_count = conf[:, f].sum()
        tp         = conf[f, f]
        fdr    = tp / pred_count * 100 if pred_count > 0 else 0.0
        recall = tp / true_count * 100 if true_count > 0 else 0.0
        fdr_list.append(fdr)
        desc = fault_desc.get(f, f"Fault {f}")
        bar  = "█" * int(recall / 5) + "░" * (20 - int(recall / 5))
        print(f"  F{str(f).zfill(2)}     {desc:<38} {fdr:>9.1f}%"
              f" {recall:>7.1f}% {true_count:>8,}")

    avg_fdr = np.mean(fdr_list)
    print("  " + "-" * 68)
    print(f"  {'AVERAGE':<48} {avg_fdr:>9.1f}%")
    print("=" * 72)

    # ── Per-intro-time detection rate ────────────────────────────────────────
    print("\n  DETECTION RATE BY FAULT INTRODUCTION TIME")
    print("  (Did the model catch faults regardless of WHEN they started?)")
    print(f"\n  {'Intro time':>12}  {'TP windows':>12}  {'Total':>8}  "
          f"{'Detection%':>12}")
    print("  " + "-" * 52)
    for t in intro_times:
        tp    = per_intro[t]["tp"]
        total = per_intro[t]["total"]
        rate  = tp / total * 100 if total > 0 else 0.0
        bar   = "█" * int(rate / 5) + "░" * (20 - int(rate / 5))
        print(f"  t = {t:>6}       {tp:>12,}  {total:>8,}  {rate:>10.1f}%  {bar}")

    print()
    print("  NOTE: If detection% is high only near t=160 and drops for other")
    print("  intro times, the model has learnt the timestamp, not the signal.")
    print("=" * 72)

    return conf, fdr_list, per_intro


# ─── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  TEP Mixed-Timing Fault Detection Evaluation")
    print("=" * 72)

    # Load data
    print(f"\nLoading {DATA_FILE} …", end=" ", flush=True)
    df_data = pd.read_csv(DATA_FILE)
    print(f"OK  ({len(df_data):,} rows)")

    print(f"Loading {META_FILE} …", end=" ", flush=True)
    df_meta = pd.read_csv(META_FILE)
    print(f"OK  ({len(df_meta)} runs)")

    # Load norm params and model
    print("\nLoading model …")
    means, stds = load_model_and_norm()

    scols     = sensor_cols(df_data)
    n_sensors = len(scols)
    model     = FaultClassifierLSTM(n_sensors=n_sensors,
                                    hidden=HIDDEN_SIZE,
                                    n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_FILE, map_location="cpu"))
    print(f"  Weights loaded from {MODEL_FILE}")

    print(f"\n  Dataset: {len(df_data):,} rows, "
          f"{len(df_meta)} runs, "
          f"{df_meta['faultNumber'].nunique()} fault types")
    print(f"  Fault intro times: "
          f"{sorted(df_meta['fault_intro_sample'].unique())}")

    conf, fdr_list, per_intro = evaluate_mixed(model, df_data, df_meta,
                                               means, stds)


if __name__ == "__main__":
    main()