"""
create_mixed_test_dataset.py  —  TEP Mixed-Fault-Timing Test Dataset Builder
=============================================================================
PURPOSE
-------
The original faulty_testing.RData always introduces faults at sample t=160.
A model that memorised "anything after t=160 is a fault" would score well
without actually learning fault signatures.

This script creates a MIXED testing dataset where each run has the fault
introduced at a DIFFERENT random time (e.g., 50, 68, 100, 200, 250, 400 …),
so you can verify the model detects faults from signal patterns, not timestamps.

HOW THE MIXING WORKS
--------------------
Original faulty_testing.RData (960 samples/run):
    samples  1 – 160  : pre-fault (normal operation)
    samples 161 – 960 : POST-FAULT  (800 confirmed-fault samples)

For each run in the mixed dataset with fault_intro = T:
    samples  1 –  T   : taken from fault_free_testing  (genuine normal)
    samples T+1 – T+800: taken from faulty_testing, post-fault portion (161→960)

  → If T ≤ 160  : total run length = T + 800  (shorter than 960)
  → If T >  160 : we only take (960-T) post-fault rows → total stays 960

The dataset is saved with a `fault_intro_sample` metadata column so the
evaluation code knows the true switch-over point for each run.

USAGE
-----
    python create_mixed_test_dataset.py

REQUIRED FILES (same folder or set paths below)
    fault_free_testing.RData    — 500 normal runs × 960 samples
    faulty_testing.RData        — 21 faults × 500 runs × 960 samples

OUTPUT
------
    mixed_test_dataset.csv          — full sensor data (big file)
    mixed_test_metadata.csv         — run_id, faultNumber, fault_intro_sample
    mixed_test_dataset_sample.csv   — first 5 runs preview
"""

import os
import random
import numpy as np
import pandas as pd
import pyreadr

# ─── CONFIG ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAULT_FREE_FILE = os.path.join(BASE_DIR, "..", "data", "raw", "fault_free_testing.RData")
FAULTY_FILE     = os.path.join(BASE_DIR, "..", "data", "raw", "faulty_testing.RData")

OUTPUT_FULL     = os.path.join(BASE_DIR, "..", "data", "processed", "mixed_test_dataset.csv")
OUTPUT_META     = os.path.join(BASE_DIR, "..", "data", "processed", "mixed_test_metadata.csv")
OUTPUT_SAMPLE   = os.path.join(BASE_DIR, "..", "data", "processed", "mixed_test_dataset_sample.csv")

# Fault intro times to rotate through (samples, 1-indexed)
# Cover early, mid, original (160), and late introductions
FAULT_INTRO_TIMES = [50, 68, 100, 130, 160, 200, 250, 300, 400]

# How many runs to create PER FAULT
# The original has 500 runs per fault; we create one run per intro_time value
# → len(FAULT_INTRO_TIMES) runs × 20 faults = 180 runs total by default
# Increase RUNS_PER_FAULT if you want more repetitions with random timing
RUNS_PER_FAULT = len(FAULT_INTRO_TIMES)   # one run per defined intro time

# Faults to include (skip 3, 9, 15 — known undetectable in TEP literature)
SKIP_FAULTS = {3, 9, 15}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ORIGINAL_FAULT_INTRO = 160   # in faulty_testing, fault always starts here
ORIGINAL_TOTAL_LEN   = 960   # samples per run in original test data


# ─── HELPERS ─────────────────────────────────────────────────────────────────
def load_rdata(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n  ✗ Missing: {path}\n"
            f"    Put it in the same folder as this script and retry."
        )
    print(f"  Loading {path} …", end=" ", flush=True)
    r  = pyreadr.read_r(path)
    df = r[list(r.keys())[0]]
    print(f"OK  ({len(df):,} rows)")
    return df


def sensor_cols(df):
    return [c for c in df.columns
            if c.startswith("xmeas_") or c.startswith("xmv_")]


# ─── MAIN ────────────────────────────────────────────────────────────────────
def build_mixed_dataset():
    print("=" * 68)
    print("  TEP Mixed-Timing Test Dataset Builder")
    print("=" * 68)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\nLoading source files …")
    df_normal = load_rdata(FAULT_FREE_FILE)
    df_faulty = load_rdata(FAULTY_FILE)

    scols = sensor_cols(df_normal)
    print(f"  Sensor columns : {len(scols)}")

    normal_runs = sorted(int(r) for r in df_normal["simulationRun"].unique())
    faulty_runs = sorted(int(r) for r in df_faulty["simulationRun"].unique())
    faults      = sorted(int(f) for f in df_faulty["faultNumber"].unique()
                         if int(f) not in SKIP_FAULTS and int(f) != 0)

    print(f"  Normal runs    : {len(normal_runs)}")
    print(f"  Fault runs     : {len(faulty_runs)}  (per fault)")
    print(f"  Faults used    : {faults}")
    print(f"  Intro times    : {FAULT_INTRO_TIMES}")

    # Pre-build lookup dicts for speed
    print("\nIndexing runs …", end=" ", flush=True)
    normal_index = {}
    for run in normal_runs:
        sub = (df_normal[df_normal["simulationRun"] == run]
               .sort_values("sample")[scols].values.astype(np.float32))
        normal_index[run] = sub    # shape (960, 52)

    faulty_index = {}
    for fault in faults:
        faulty_index[fault] = {}
        for run in faulty_runs:
            sub = (df_faulty[(df_faulty["faultNumber"] == fault) &
                             (df_faulty["simulationRun"] == run)]
                   .sort_values("sample")[scols].values.astype(np.float32))
            if len(sub) > 0:
                faulty_index[fault][run] = sub
    print("Done.")

    # ── Build mixed runs ──────────────────────────────────────────────────────
    print(f"\nBuilding mixed dataset …")
    rows_list  = []   # will hold dicts → DataFrame rows (chunked for memory)
    meta_rows  = []
    new_run_id = 0

    # We cycle normal runs to pair with (fault, intro_time) combos
    normal_run_cycle = normal_runs.copy()
    random.shuffle(normal_run_cycle)
    norm_idx = 0

    for fault in faults:
        # Shuffle the intro times so each fault gets different orderings
        intro_schedule = FAULT_INTRO_TIMES.copy()
        # If RUNS_PER_FAULT > len(FAULT_INTRO_TIMES), add random extras
        while len(intro_schedule) < RUNS_PER_FAULT:
            intro_schedule.append(random.choice(FAULT_INTRO_TIMES))

        avail_faulty_runs = sorted(faulty_index[fault].keys())
        random.shuffle(avail_faulty_runs)

        for i, T_fault in enumerate(intro_schedule):
            new_run_id += 1

            # Pick a normal run and a faulty run (cycle through available)
            norm_run   = normal_run_cycle[norm_idx % len(normal_run_cycle)]
            faulty_run = avail_faulty_runs[i % len(avail_faulty_runs)]
            norm_idx  += 1

            normal_data = normal_index[norm_run]           # (960, 52)
            faulty_data = faulty_index[fault][faulty_run]  # (960, 52)

            # POST-fault portion from original faulty run
            # original fault intro = 160 (1-indexed), so index = 160 (0-indexed)
            post_fault_data = faulty_data[ORIGINAL_FAULT_INTRO:]   # (800, 52)

            # ── PRE-FAULT portion ─────────────────────────────────────────
            pre_fault_data = normal_data[:T_fault]       # (T_fault, 52)

            # ── POST-FAULT portion to append ──────────────────────────────
            if T_fault <= ORIGINAL_FAULT_INTRO:
                # Take all 800 post-fault samples → run length = T + 800
                used_post = post_fault_data              # (800, 52)
            else:
                # Take only enough to keep total at 960
                n_post = ORIGINAL_TOTAL_LEN - T_fault    # < 800
                used_post = post_fault_data[:n_post]     # (n_post, 52)

            # ── Stitch ────────────────────────────────────────────────────
            combined = np.vstack([pre_fault_data, used_post])  # (total, 52)
            total_len = len(combined)

            # Build rows for this run
            for s_idx in range(total_len):
                row = {
                    "faultNumber"        : fault,
                    "simulationRun"      : new_run_id,
                    "sample"             : s_idx + 1,       # 1-indexed
                    "fault_intro_sample" : T_fault,
                    "source_normal_run"  : norm_run,
                    "source_faulty_run"  : faulty_run,
                }
                for ci, col in enumerate(scols):
                    row[col] = combined[s_idx, ci]
                rows_list.append(row)

            meta_rows.append({
                "simulationRun"      : new_run_id,
                "faultNumber"        : fault,
                "fault_intro_sample" : T_fault,
                "total_samples"      : total_len,
                "source_normal_run"  : norm_run,
                "source_faulty_run"  : faulty_run,
            })

        print(f"  Fault {fault:02d} : {RUNS_PER_FAULT} runs  "
              f"(intro times: {intro_schedule})")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\nConverting to DataFrame …", end=" ", flush=True)
    df_mixed = pd.DataFrame(rows_list)
    df_meta  = pd.DataFrame(meta_rows)
    print(f"Done.  Shape: {df_mixed.shape}")

    print(f"\nSaving {OUTPUT_FULL} …", end=" ", flush=True)
    df_mixed.to_csv(OUTPUT_FULL, index=False)
    print(f"OK  ({os.path.getsize(OUTPUT_FULL) / 1e6:.1f} MB)")

    print(f"Saving {OUTPUT_META} …", end=" ", flush=True)
    df_meta.to_csv(OUTPUT_META, index=False)
    print("OK")

    # Sample preview (first 5 runs)
    sample_runs = df_meta["simulationRun"].head(5).tolist()
    df_sample   = df_mixed[df_mixed["simulationRun"].isin(sample_runs)]
    df_sample.to_csv(OUTPUT_SAMPLE, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  MIXED DATASET SUMMARY")
    print("=" * 68)
    print(f"  Total rows       : {len(df_mixed):,}")
    print(f"  Total runs       : {new_run_id}")
    print(f"  Faults covered   : {faults}")
    print(f"  Intro times used : {FAULT_INTRO_TIMES}")
    print()
    print("  Per-fault run count and intro-time distribution:")
    for fault in faults:
        sub = df_meta[df_meta["faultNumber"] == fault]
        intro_dist = sub["fault_intro_sample"].value_counts().sort_index().to_dict()
        print(f"    F{fault:02d} : {len(sub)} runs — intro times: {intro_dist}")

    print(f"\n  Outputs written:")
    print(f"    {OUTPUT_FULL}           ← feed to your evaluation code")
    print(f"    {OUTPUT_META}       ← run-level metadata (fault intro times)")
    print(f"    {OUTPUT_SAMPLE}  ← quick preview of first 5 runs")
    print("=" * 68)

    return df_mixed, df_meta


if __name__ == "__main__":
    build_mixed_dataset()
