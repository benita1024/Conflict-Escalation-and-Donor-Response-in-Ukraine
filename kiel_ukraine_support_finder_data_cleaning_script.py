"""
Kiel Ukraine Support Tracker - Data Cleaning Script
====================================================
Input:  Ukraine_Support_Tracker_Release_27.xlsx
Output: kiel_panel_clean.csv

Final panel structure:
  - One row per: donor × year_month × aid_type
  - Value column: aid_eur_m (EUR millions, allocations)
  - 41 bilateral donors × 36 months (Feb 2022–Jan 2025) × 3 aid types
  - Expected: 4,428 rows (fully balanced, zeros for missing cells)
  - Period trimmed to match ACLED data availability (ends Feb 2025)
  - aid_type_specific preserved as pipe-delimited list of subcategories per cell

Key design decisions (documented here for methods section):
  1. We use ALLOCATIONS (not Commitments). In the Kiel data, ~95% of rows
     are coded as 'Allocation' — these are aid packages that have been
     delivered or formally earmarked for delivery. 'Commitment' rows (n=275)
     are a small set of separately tracked pledges with NaN values.
     Allocations are the standard measure used in all published Kiel analyses.

  2. We use 'tot_sub_activity_value_EUR_redistr' as the value column.
     This is Kiel's preferred EUR column: it handles sub-activity breakdowns
     (e.g., individual weapons within a package), converts all currencies to EUR,
     and redistributes values for items with unknown pricing. It exactly
     replicates the published country-level totals in the Kiel tracker.

  3. Rows with month=0 (unknown announcement date) are EXCLUDED from the main
     panel and set to zero in the balanced panel. These account for ~5% of
     total value and represent multi-year packages without traceable month.
     This is conservative and standard in time-series aid research.

  4. Supranational donors (EU Commission, EIB, European Peace Facility) are
     EXCLUDED. We study bilateral government decisions, and supranational
     commitments follow different political economy mechanisms.

Author: Benita Besa
Course: Econ 590, Duke University
"""

import pandas as pd
import numpy as np

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────
INPUT_FILE  = "Ukraine_Support_Tracker_Release_27.xlsx"   # rename as needed
OUTPUT_FILE = "kiel_panel_clean.csv"
SHEET       = "Bilateral Assistance, MAIN DATA"

# 41 bilateral donor countries we keep
BILATERAL_DONORS = [
    "Australia", "Austria", "Belgium", "Bulgaria", "Canada",
    "China", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Iceland", "India", "Ireland", "Italy",
    "Japan", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "New Zealand", "Norway", "Poland", "Portugal",
    "Romania", "Slovakia", "Slovenia", "South Korea", "Spain",
    "Sweden", "Switzerland", "Taiwan", "Turkiye",
    "United Kingdom", "United States",
]

# Donor group classification for heterogeneity analysis (H2)
DONOR_GROUPS = {
    "frontline_adjacent": [
        "Estonia", "Latvia", "Lithuania", "Poland", "Finland", "Czechia",
    ],
    "major_western": [
        "United States", "United Kingdom", "Germany", "France",
    ],
    "reluctant": [
        "Italy", "Spain", "Hungary", "Slovakia",
    ],
    # Significant Western contributors outside the core four
    "other_western": [
        "Canada", "Australia", "Japan", "Norway", "Netherlands", "Sweden",
    ],
    # All remaining donors receive "other"
}
donor_group_map = {
    donor: group
    for group, donors in DONOR_GROUPS.items()
    for donor in donors
}

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading …")
raw = pd.read_excel(INPUT_FILE, sheet_name=SHEET)
print(f"  Raw: {raw.shape[0]:,} rows × {raw.shape[1]} columns")

# ── 2. FORCE NUMERIC ON VALUE COLUMNS ────────────────────────────────────────
# Some cells contain "." as a placeholder in the original Excel — coerce to NaN
VALUE_COL = "tot_sub_activity_value_EUR_redistr"
for col in ["tot_activity_value_EUR", "tot_sub_activity_value_EUR", VALUE_COL]:
    raw[col] = pd.to_numeric(raw[col], errors="coerce")

# ── 3. KEEP ONLY ALLOCATION ROWS FOR BILATERAL DONORS ────────────────────────
df = raw[
    (raw["measure"] == "Allocation") &
    (raw["donor"].isin(BILATERAL_DONORS))
].copy()
print(f"  After bilateral-donor + allocation filter: {len(df):,} rows")

# Clean up aid_type_general (one entry has a trailing space: "Humanitarian ")
df["aid_type_general"] = df["aid_type_general"].astype(str).str.strip()

# ── 4. DECODE THE MONTH COLUMN ────────────────────────────────────────────────
# Kiel's 'month' is an integer offset from December 2021:
#   0 → unknown/multi-period (excluded from time panel)
#   1 → January 2022
#   2 → February 2022
#   …
#  48 → December 2025

REFERENCE = pd.Period("2021-12", freq="M")

df["year_month"] = df["month"].apply(
    lambda m: str(REFERENCE + int(m)) if pd.notna(m) and int(m) != 0 else np.nan
)

n_unknown = df["year_month"].isna().sum()
pct_unknown = n_unknown / len(df) * 100
val_unknown = df.loc[df["year_month"].isna(), VALUE_COL].sum() / 1e9
print(f"  Unknown-month rows: {n_unknown} ({pct_unknown:.1f}%) "
      f"→ €{val_unknown:.1f}B excluded from time panel")

# Keep only rows with known month
df = df.dropna(subset=["year_month"]).copy()

# ── 5. RESTRICT TO WAR PERIOD ─────────────────────────────────────────────────
df["period"] = pd.PeriodIndex(df["year_month"], freq="M")
df = df[
    (df["period"] >= pd.Period("2022-02", freq="M")) &
    (df["period"] <= pd.Period("2025-01", freq="M"))
].copy()
print(f"  After war-period filter (Feb 2022 – Jan 2025): {len(df):,} rows")

# ── 6. AGGREGATE TO DONOR × MONTH × AID_TYPE ──────────────────────────────────
# Sum all sub-activity rows within each cell
# (negative values = corrections/cancellations; keep them so they net out correctly)

# Clean aid_type_specific
df["aid_type_specific"] = df["aid_type_specific"].astype(str).str.strip()

# Aggregate value
panel_val = (
    df
    .groupby(["donor", "year_month", "aid_type_general"], as_index=False)
    [VALUE_COL]
    .sum()
    .rename(columns={VALUE_COL: "aid_eur_m"})
)
panel_val["aid_eur_m"] = panel_val["aid_eur_m"] / 1e6   # convert to millions

# Collect unique aid_type_specific subcategories per cell as pipe-delimited string
panel_spec = (
    df
    .groupby(["donor", "year_month", "aid_type_general"])["aid_type_specific"]
    .apply(lambda x: " | ".join(sorted(set(x.dropna()))))
    .reset_index()
)

panel = panel_val.merge(panel_spec, on=["donor", "year_month", "aid_type_general"], how="left")
print(f"  Aggregated: {len(panel):,} non-zero donor×month×type cells")

# ── 7. BUILD BALANCED PANEL (FILL ZEROS FOR MISSING CELLS) ────────────────────
# Many donor-month-type combinations have zero aid — we need them as explicit
# zeros for the distributed lag regression to work correctly.

all_months = [str(REFERENCE + i) for i in range(2, 38)]   # Feb 2022–Jan 2025 = 36 months
all_types  = ["Military", "Financial", "Humanitarian"]

full_index = pd.MultiIndex.from_product(
    [BILATERAL_DONORS, all_months, all_types],
    names=["donor", "year_month", "aid_type_general"]
)
# Reindex numeric columns only (fill_value=0.0), then re-attach aid_type_specific
panel_reindexed = (
    panel[["donor", "year_month", "aid_type_general", "aid_eur_m"]]
    .set_index(["donor", "year_month", "aid_type_general"])
    .reindex(full_index, fill_value=0.0)
    .reset_index()
)
# Re-attach aid_type_specific (empty string for zero-fill rows)
panel = panel_reindexed.merge(
    panel[["donor", "year_month", "aid_type_general", "aid_type_specific"]],
    on=["donor", "year_month", "aid_type_general"],
    how="left"
)
panel["aid_type_specific"] = panel["aid_type_specific"].fillna("")
expected = len(BILATERAL_DONORS) * len(all_months) * len(all_types)
assert len(panel) == expected, f"Expected {expected} rows, got {len(panel)}"
print(f"  Balanced panel: {len(panel):,} rows "
      f"({len(BILATERAL_DONORS)} donors × {len(all_months)} months × {len(all_types)} types = {len(BILATERAL_DONORS)*len(all_months)*len(all_types):,} rows)")

# ── 8. ADD ANALYTICAL COLUMNS ─────────────────────────────────────────────────

# Donor group for heterogeneity analysis
panel["donor_group"] = panel["donor"].map(donor_group_map).fillna("other")

# Binary flags
panel["frontline_flag"] = (panel["donor_group"] == "frontline_adjacent").astype(int)
panel["reluctant_flag"]  = (panel["donor_group"] == "reluctant").astype(int)

# Time variables
panel["period_obj"]   = pd.PeriodIndex(panel["year_month"], freq="M")
panel["year"]         = panel["period_obj"].dt.year
panel["month_num"]    = panel["period_obj"].dt.month   # 1–12

# Months since invasion (Feb 2022 = 1)
invasion_start = pd.Period("2022-02", freq="M")
panel["t"] = panel["period_obj"].apply(lambda p: (p - invasion_start).n + 1)

# Post-2022 flag (for aid-fatigue robustness check)
# Jan 2023 onward = war is protracted; initial mobilization shock has passed
panel["post_2022"] = (panel["year"] >= 2023).astype(int)

# Winter flag: Oct–Mar (Russia's energy campaign seasons)
panel["winter"] = panel["month_num"].isin([10, 11, 12, 1, 2, 3]).astype(int)

# Drop period helper column
panel = panel.drop(columns=["period_obj"])

# ── 9. SANITY CHECKS ──────────────────────────────────────────────────────────
print("\n── SANITY CHECKS ──")

# Check published Kiel totals (from Country Summary sheet)
# Expected approx: Military ~162B, Financial ~158B, Humanitarian ~24B (allocations only)
totals = panel.groupby("aid_type_general")["aid_eur_m"].sum() / 1e3   # → EUR billion
print("\nTotal allocations by aid type (€ billion):")
print(totals.round(1))
print("  [Expected approx: Military ~162, Financial ~158, Humanitarian ~24]")

# Check a known large donor
us_total = panel[panel["donor"] == "United States"]["aid_eur_m"].sum() / 1e3
print(f"\nUS total allocations: €{us_total:.1f}B  [Expected: ~115B]")

# Check balance
assert len(panel) == expected
print(f"\nBalance check passed: {len(panel):,} rows ✓")

# Check no nulls in key columns
for col in ["donor", "year_month", "aid_type_general", "aid_eur_m",
            "donor_group", "t"]:
    assert panel[col].isna().sum() == 0, f"NaN found in {col}"
print("Null check passed: no nulls in key columns ✓")

# Check t range
assert panel["t"].min() == 1 and panel["t"].max() == 36, "t range error"
print(f"Time range check passed: t = 1 (Feb 2022) to 36 (Jan 2025) ✓")

# ── 10. PREVIEW ───────────────────────────────────────────────────────────────
print("\n── PANEL PREVIEW ──")
print("\nDonor group counts:")
print(panel.drop_duplicates("donor")["donor_group"].value_counts())

print("\nSample rows (non-zero aid, sorted by value):")
preview = (
    panel[panel["aid_eur_m"] > 0]
    .sort_values("aid_eur_m", ascending=False)
    .head(10)
    [["donor", "year_month", "aid_type_general", "aid_eur_m", "donor_group", "t"]]
)
print(preview.to_string(index=False))

print("\nColumn descriptions:")
col_desc = {
    "donor":            "Donor country name",
    "year_month":       "Year-month string (YYYY-MM)",
    "aid_type_general": "Aid type: Military | Financial | Humanitarian",
    "aid_eur_m":        "Aid allocation in EUR millions (0 if no aid that month)",
    "donor_group":      "frontline_adjacent | major_western | reluctant | other_western | other",
    "frontline_flag":   "1 if frontline_adjacent donor",
    "reluctant_flag":   "1 if reluctant donor",
    "year":             "Calendar year",
    "month_num":        "Calendar month (1–12)",
    "t":                "Months since invasion (Feb 2022 = 1)",
    "aid_type_specific":"Pipe-delimited subcategories within aid_type_general",
    "post_2022":        "1 if year >= 2023 (war is protracted; aid fatigue check)",
    "winter":           "1 if Oct–Mar (Russia energy campaign season)",
}
for col, desc in col_desc.items():
    print(f"  {col:<22} {desc}")

# ── 11. SORT ──────────────────────────────────────────────────────────────────
# Sort by date first, then donor and aid_type within each month
panel = panel.sort_values(
    ["year_month", "donor", "aid_type_general"]
).reset_index(drop=True)

# ── 12. SAVE ──────────────────────────────────────────────────────────────────
panel.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved: {OUTPUT_FILE}")
print(f"Final shape: {panel.shape}")