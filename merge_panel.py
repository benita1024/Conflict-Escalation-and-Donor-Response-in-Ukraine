"""
Merge Script: ACLED + Kiel Ukraine Support Tracker
===================================================
Input:  acled_monthly_clean.csv   (36 monthly conflict rows, Feb 2022 – Jan 2025)
        kiel_panel_clean.csv      (4,428 rows: 41 donors × 36 months × 3 aid types)
Output: panel_merged.csv          (4,428 rows with conflict variables attached)

Merge logic:
  - ACLED is month-level; Kiel is donor × month × aid_type
  - Every Kiel row in a given month gets the same conflict counts for that month
  - Conflict is national-level so this is correct by design
  - Pre-war rows (2021-12, 2022-01) in ACLED are excluded from the merge

Author: Benita Besa, Mingyuan Song
Course: Econ 590, Duke University
"""

import pandas as pd
import numpy as np

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────
ACLED_FILE  = "../data/acled_monthly_clean.csv"
KIEL_FILE   = "../data/kiel_panel_clean.csv"
OUTPUT_FILE = "../data/panel_merged.csv"

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading datasets …")
acled = pd.read_csv(ACLED_FILE)
kiel  = pd.read_csv(KIEL_FILE)
print(f"  ACLED: {acled.shape[0]} rows × {acled.shape[1]} cols")
print(f"  Kiel:  {kiel.shape[0]} rows × {kiel.shape[1]} cols")

# ── 2. FILTER ACLED TO WAR PERIOD ONLY ───────────────────────────────────────
# Drop pre-war rows (Dec 2021, Jan 2022) — not in Kiel panel
acled = acled[acled['year_month'] >= '2022-02'].copy()
print(f"  ACLED after war-period filter: {len(acled)} rows "
      f"({acled['year_month'].min()} to {acled['year_month'].max()})")

# ── 3. CHECK MONTH ALIGNMENT ──────────────────────────────────────────────────
acled_months = set(acled['year_month'])
kiel_months  = set(kiel['year_month'])
only_acled   = acled_months - kiel_months
only_kiel    = kiel_months  - acled_months

if only_acled:
    print(f"  WARNING: months in ACLED but not Kiel: {sorted(only_acled)}")
if only_kiel:
    print(f"  WARNING: months in Kiel but not ACLED: {sorted(only_kiel)}")
if not only_acled and not only_kiel:
    print(f"  Month alignment check passed: both datasets cover "
          f"{acled['year_month'].min()} to {acled['year_month'].max()} ✓")

# ── 4. PREPARE ACLED FOR MERGE ────────────────────────────────────────────────
# Drop 't' from ACLED — Kiel already has it, avoid duplicate columns
acled_merge = acled.drop(columns=['t'])

# ── 5. MERGE ──────────────────────────────────────────────────────────────────
# Left join: keep all Kiel rows, attach conflict data by month
print("\nMerging …")
panel = kiel.merge(acled_merge, on='year_month', how='left')
print(f"  Merged shape: {panel.shape}")

# Verify no rows were added or dropped
assert len(panel) == len(kiel), \
    f"Row count changed after merge: expected {len(kiel)}, got {len(panel)}"
print(f"  Row count check passed: {len(panel):,} rows ✓")

# Verify no NaN in conflict columns (every Kiel month should have ACLED data)
conflict_cols = ['infrastructure_attacks', 'battlefield_events', 'infra_spike']
for col in conflict_cols:
    n_null = panel[col].isna().sum()
    assert n_null == 0, f"NaN found in {col}: {n_null} nulls"
print(f"  Null check passed: no nulls in conflict columns ✓")

# ── 6. ADD LAG VARIABLES ──────────────────────────────────────────────────────
# Lags 1–6 of both treatment variables, computed within each donor × aid_type group
# Sort by donor, aid_type, then time so shift() works correctly
print("\nComputing lag variables …")
panel = panel.sort_values(
    ['donor', 'aid_type_general', 'year_month']
).reset_index(drop=True)

for lag in range(1, 7):
    panel[f'infra_lag{lag}']  = (panel
        .groupby(['donor', 'aid_type_general'])['infrastructure_attacks']
        .shift(lag))
    panel[f'battle_lag{lag}'] = (panel
        .groupby(['donor', 'aid_type_general'])['battlefield_events']
        .shift(lag))

print(f"  Added 12 lag columns (infra_lag1–6, battle_lag1–6)")

# Check lag columns — first 6 rows of one group should have NaN lags
sample = panel[
    (panel['donor'] == 'Germany') &
    (panel['aid_type_general'] == 'Military')
][['year_month', 'infrastructure_attacks', 'infra_lag1', 'infra_lag2', 'infra_lag6']].head(8)
print(f"\n  Lag check (Germany Military):")
print(sample.to_string(index=False))

# ── 7. FINAL SORT: by date → donor → aid_type ─────────────────────────────────
panel = panel.sort_values(
    ['year_month', 'donor', 'aid_type_general']
).reset_index(drop=True)

# ── 8. SUMMARY ────────────────────────────────────────────────────────────────
print(f"\n── FINAL PANEL SUMMARY ──")
print(f"  Shape:   {panel.shape}")
print(f"  Period:  {panel['year_month'].min()} to {panel['year_month'].max()}")
print(f"  Donors:  {panel['donor'].nunique()}")
print(f"  Months:  {panel['year_month'].nunique()}")
print(f"  Aid types: {sorted(panel['aid_type_general'].unique())}")
print(f"  Spike months: {sorted(panel[panel['infra_spike']==1]['year_month'].unique())}")
print(f"\n  Columns:")
for col in panel.columns:
    print(f"    {col}")

# ── 9. SAVE ───────────────────────────────────────────────────────────────────
panel.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")