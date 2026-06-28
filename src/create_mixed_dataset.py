"""
create_mixed_dataset.py  —  TEP Blind Mixed Testing Dataset Generator
======================================================================
Creates a 40-run blind dataset for the TEP anomaly detection benchmark.

  20 faults × 2 runs each = 40 runs, shuffled so run numbers reveal nothing.

OUTPUT
──────
  mixed_blind_testing.RData   → load into the server  (share freely)
  mixed_blind_key.json        → contains the answers   ⚠ KEEP PRIVATE

USAGE
─────
  python create_mixed_dataset.py

SOURCE PRIORITY
───────────────
  1. faulty_testing.RData  — Harvard Dataverse TEP testing file (preferred)
                             doi:10.7910/DVN/6C3JR1  → TEP_Faulty_Testing.RData
                             Rename to faulty_testing.RData
  2. faulty_training.RData — Training file (falls back to validation runs 451-500)

Place the source file in the same folder and run this script.
"""

import os, sys, json, random
import numpy as np
import pandas as pd
import pyreadr

# ── BASE DIR (all paths resolved relative to this script, not the terminal CWD) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── CONFIG ────────────────────────────────────────────────────────────────────
TESTING_FILE  = os.path.join(BASE_DIR, "..", "data", "raw", "faulty_testing.RData")     # Harvard Dataverse (preferred)
TRAINING_FILE = os.path.join(BASE_DIR, "..", "data", "raw", "faulty_training.RData")    # Fallback

OUTPUT_RDATA  = os.path.join(BASE_DIR, "..", "data", "raw", "mixed_blind_testing.RData")
OUTPUT_KEY    = os.path.join(BASE_DIR, "..", "data", "processed", "mixed_blind_key.json")

FAULTS          = list(range(1, 21))   # faults 1–20
RUNS_PER_FAULT  = 2                    # 2 × 20 = 40 blind runs
SENTINEL_FAULT  = 99                   # placeholder faultNumber in output
                                       # (hides the true label from the file)

# When falling back to training data, use only the validation split
# (runs 451-500) so these are fresh, never-trained-on simulations
TRAINING_VAL_RUNS = set(range(451, 501))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def hr(char="─", n=65):
    print(char * n)

def load_source():
    """Load source RData. Prefer testing file, fall back to training."""
    if os.path.exists(TESTING_FILE):
        print(f"  Source : {TESTING_FILE}  (Harvard Dataverse testing set)")
        r  = pyreadr.read_r(TESTING_FILE)
        df = r[list(r.keys())[0]]
        print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
        avail_fn = lambda fault: sorted(
            int(x) for x in df[df["faultNumber"] == fault]["simulationRun"].unique()
        )
        return df, avail_fn, "testing"

    if os.path.exists(TRAINING_FILE):
        print(f"  Source : {TRAINING_FILE}  (training file — validation split)")
        r  = pyreadr.read_r(TRAINING_FILE)
        df = r[list(r.keys())[0]]
        print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} cols")
        avail_fn = lambda fault: sorted(
            int(x) for x in df[
                (df["faultNumber"] == fault) &
                (df["simulationRun"].isin(TRAINING_VAL_RUNS))
            ]["simulationRun"].unique()
        )
        print(f"  Runs   : using validation split {min(TRAINING_VAL_RUNS)}–{max(TRAINING_VAL_RUNS)}")
        return df, avail_fn, "training"

    print("\n⚠  ERROR: No source file found.")
    print(f"   Expected one of:")
    print(f"     {TESTING_FILE}")
    print(f"     {TRAINING_FILE}")
    print("   Download faulty_testing.RData from:")
    print("   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1")
    print("   and rename to  faulty_testing.RData  in the SAME folder as this script.")
    print("   — OR — ensure faulty_training.RData is present in the same folder.")
    sys.exit(1)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    hr("═")
    print("  TEP Blind Mixed Testing Dataset Generator")
    hr("═")

    print("\nLoading source data …")
    df_source, avail_fn, mode = load_source()

    # ── Select 2 runs per fault ────────────────────────────────────────────
    print(f"\nSelecting {RUNS_PER_FAULT} runs per fault  ({len(FAULTS)} faults × {RUNS_PER_FAULT} = {len(FAULTS)*RUNS_PER_FAULT} total) …")
    assignments = []   # list of (true_fault, source_run)

    for fault in FAULTS:
        runs = avail_fn(fault)
        if len(runs) < RUNS_PER_FAULT:
            print(f"  ⚠  Fault {fault:02d}: only {len(runs)} run(s) available — need {RUNS_PER_FAULT}")
            sys.exit(1)
        chosen = random.sample(runs, RUNS_PER_FAULT)
        for r in chosen:
            assignments.append((fault, r))
        print(f"  Fault {fault:02d}: picked runs {chosen}")

    # ── Shuffle → blind run numbers 1–40 ──────────────────────────────────
    random.shuffle(assignments)

    # ── Build key (ANSWER KEY — keep private) ─────────────────────────────
    key = {}
    for blind_run, (true_fault, source_run) in enumerate(assignments, start=1):
        key[str(blind_run)] = {
            "true_fault":  int(true_fault),
            "source_run":  int(source_run),
            "source_mode": mode,
        }

    hr()
    print("Blind run → true fault mapping  ⚠ KEEP THIS SECRET ⚠")
    hr()
    for br in range(1, len(key) + 1):
        info = key[str(br)]
        print(f"  Run {br:>2} →  Fault {info['true_fault']:02d}  (source run {info['source_run']})")
    hr()

    # ── Build output DataFrame ─────────────────────────────────────────────
    print("\nBuilding output DataFrame …")
    frames = []
    for blind_run, (true_fault, source_run) in enumerate(assignments, start=1):
        chunk = df_source[
            (df_source["faultNumber"]   == true_fault) &
            (df_source["simulationRun"] == source_run)
        ].copy().sort_values("sample").reset_index(drop=True)

        chunk["faultNumber"]   = SENTINEL_FAULT   # ← hides true fault
        chunk["simulationRun"] = blind_run         # ← new shuffled run ID

        frames.append(chunk)
        print(f"  Blind run {blind_run:>2}  ←  Fault {true_fault:02d}, source run {source_run}  "
              f"({len(chunk)} samples)")

    df_out = pd.concat(frames, ignore_index=True)

    print(f"\nOutput DataFrame  shape : {df_out.shape}")
    print(f"  faultNumber  values : {sorted(df_out['faultNumber'].unique())}")
    print(f"  simulationRun values : {sorted(df_out['simulationRun'].unique())}")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving {OUTPUT_RDATA} …")
    pyreadr.write_rdata(OUTPUT_RDATA, df_out, df_name="mixed_blind_testing")
    size_mb = os.path.getsize(OUTPUT_RDATA) / 1e6
    print(f"  ✓  {OUTPUT_RDATA}  ({size_mb:.1f} MB)")

    print(f"Saving {OUTPUT_KEY} …")
    with open(OUTPUT_KEY, "w") as f:
        json.dump(key, f, indent=2)
    print(f"  ✓  {OUTPUT_KEY}  ({len(key)} entries)")

    hr("═")
    print("  DONE")
    hr("═")
    print(f"""
  ✓  {OUTPUT_RDATA}
  ✓  {OUTPUT_KEY}   ⚠ KEEP PRIVATE

  Both files were saved next to this script ({BASE_DIR}).
  Place them in the same folder as tep_server_v5_blind.py
  (they already are if the server is in the same folder).

  The .RData file has faultNumber=99 for every row — the true fault
  is hidden.  The server reads the key file to reveal answers only
  after the AI has made its prediction.

  Run numbers 1–40 are in a random shuffled order.  You won't know
  which fault is which just by looking at the run number.  The AI's
  job is to classify each run correctly.
""")

if __name__ == "__main__":
    main()
