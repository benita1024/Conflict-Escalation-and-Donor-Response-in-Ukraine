"""
Donor Heterogeneity Analysis
=============================
Re-estimates the 6-lag distributed lag panel regression separately for:
  - frontline_adjacent: Czechia, Estonia, Finland, Latvia, Lithuania, Poland
  - major_western:      France, Germany, United Kingdom, United States
  - reluctant:          Hungary, Italy, Slovakia, Spain

All three aid types estimated for each group.

Note on standard errors: with only 4-6 donors per group, clustered SEs at
the donor level are used but should be interpreted cautiously — small cluster
count (< 20) can make clustered SEs unreliable. Results flagged accordingly.

Input:  acled_monthly_clean.csv + kiel_panel_clean.csv
Output: donor_heterogeneity_results.csv
        donor_heterogeneity_summary.txt

Author: Benita Besa, Mingyuan Song
Course: Econ 590, Duke University
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────
N_LAGS       = 6
DONOR_GROUPS = ['frontline_adjacent', 'major_western', 'reluctant']
AID_TYPES    = ['Military', 'Financial', 'Humanitarian']
OUTPUT_CSV   = '../results/donor_heterogeneity_results.csv'
OUTPUT_TXT   = '../results/donor_heterogeneity_summary.txt'

GROUP_LABELS = {
    'frontline_adjacent': 'Frontline (CZ, EE, FI, LV, LT, PL)',
    'major_western':      'Major Western (FR, DE, UK, US)',
    'reluctant':          'Reluctant (HU, IT, SK, ES)',
}

# ── 1. LOAD AND BUILD PANEL ───────────────────────────────────────────────────
print("Loading data ...")
acled = pd.read_csv('../data/acled_monthly_clean.csv')
kiel  = pd.read_csv('../data/kiel_panel_clean.csv')

acled = acled[acled['year_month'] >= '2022-02'].copy()
panel = kiel.merge(acled.drop(columns=['t']), on='year_month', how='left')

panel = panel.sort_values(
    ['donor', 'aid_type_general', 'year_month']
).reset_index(drop=True)

for lag in range(1, N_LAGS + 1):
    panel[f'infra_lag{lag}']  = panel.groupby(
        ['donor', 'aid_type_general'])['infrastructure_attacks'].shift(lag)
    panel[f'battle_lag{lag}'] = panel.groupby(
        ['donor', 'aid_type_general'])['battlefield_events'].shift(lag)

print(f"  Panel shape: {panel.shape}")

# Print group compositions
for group in DONOR_GROUPS:
    donors = sorted(panel[panel['donor_group'] == group]['donor'].unique())
    print(f"  {group}: {donors}")

# ── 2. REGRESSION FUNCTION ────────────────────────────────────────────────────
def run_regression(df, donor_group, aid_type, n_lags):
    lag_cols = ([f'infra_lag{l}'  for l in range(1, n_lags+1)] +
                [f'battle_lag{l}' for l in range(1, n_lags+1)])

    sub = df[
        (df['donor_group']      == donor_group) &
        (df['aid_type_general'] == aid_type)
    ].dropna(subset=lag_cols).copy()

    n_donors = sub['donor'].nunique()

    if len(sub) == 0:
        print(f"    [{donor_group} | {aid_type}] No data — skipping")
        return None, None

    sub['date'] = pd.to_datetime(sub['year_month'])
    sub = sub.set_index(['donor', 'date'])

    month_dummies = pd.get_dummies(
        sub['month_num'], prefix='month', drop_first=True
    ).astype(float)
    sub = pd.concat([sub, month_dummies], axis=1)

    infra_terms  = ' + '.join([f'infra_lag{l}'  for l in range(1, n_lags+1)])
    battle_terms = ' + '.join([f'battle_lag{l}' for l in range(1, n_lags+1)])
    month_terms  = ' + '.join(month_dummies.columns.tolist())
    formula = (f'aid_eur_m ~ {infra_terms} + {battle_terms} '
               f'+ {month_terms} + EntityEffects')

    model  = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    tidy = pd.DataFrame({
        'donor_group': donor_group,
        'aid_type':    aid_type,
        'n_donors':    n_donors,
        'variable':    result.params.index,
        'coef':        result.params.values,
        'std_err':     result.std_errors.values,
        'p_value':     result.pvalues.values,
        'ci_lower':    result.params.values - 1.96 * result.std_errors.values,
        'ci_upper':    result.params.values + 1.96 * result.std_errors.values,
    })
    tidy['sig'] = tidy['p_value'].apply(
        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    )
    return result, tidy

# ── 3. RUN ALL COMBINATIONS ───────────────────────────────────────────────────
print("\nRunning donor heterogeneity regressions ...")
all_results   = []
summary_lines = []

for donor_group in DONOR_GROUPS:
    print(f"\n  === {GROUP_LABELS[donor_group]} ===")
    for aid_type in AID_TYPES:
        result, tidy = run_regression(panel, donor_group, aid_type, N_LAGS)
        if tidy is None:
            continue
        all_results.append(tidy)

        infra_cum  = tidy[tidy['variable'].isin(
            [f'infra_lag{l}'  for l in range(1, N_LAGS+1)])]['coef'].sum()
        battle_cum = tidy[tidy['variable'].isin(
            [f'battle_lag{l}' for l in range(1, N_LAGS+1)])]['coef'].sum()

        print(f"    [{aid_type}] N={result.nobs}  "
              f"R2={result.rsquared:.4f}  "
              f"Infra cumul={infra_cum:.3f}  "
              f"Battle cumul={battle_cum:.3f}")

        summary_lines.append(
            f"\n{'='*60}\n{donor_group.upper()} | {aid_type.upper()}\n{'='*60}"
        )
        summary_lines.append(str(result.summary))

results_df = pd.concat(all_results, ignore_index=True)

# ── 4. COMPARISON TABLES ──────────────────────────────────────────────────────
output_text = []
output_text.append("DONOR HETEROGENEITY ANALYSIS — 6-LAG MODEL")
output_text.append("Coefficients shown as: coef*** (p-val)")
output_text.append("Significance: *** p<0.01  ** p<0.05  * p<0.10")
output_text.append("WARNING: Small cluster count (4-6 donors) — interpret SEs cautiously")

key_vars = ([f'infra_lag{l}'  for l in range(1, N_LAGS+1)] +
            [f'battle_lag{l}' for l in range(1, N_LAGS+1)])

for aid_type in AID_TYPES:
    output_text.append(f"\n{'='*72}")
    output_text.append(f"AID TYPE: {aid_type.upper()}")
    output_text.append(f"{'='*72}")

    col_labels = [GROUP_LABELS[g].split(' ')[0] for g in DONOR_GROUPS]
    output_text.append(
        f"\n  {'Variable':<15} {'Frontline':>18} {'Major Western':>18} {'Reluctant':>18}"
    )
    output_text.append(f"  {'-'*72}")

    for var in key_vars:
        row = f"  {var:<15}"
        for donor_group in DONOR_GROUPS:
            sub = results_df[
                (results_df['donor_group'] == donor_group) &
                (results_df['aid_type']    == aid_type) &
                (results_df['variable']    == var)
            ]
            if len(sub) == 0:
                row += f"  {'n/a':>16}"
            else:
                coef = sub['coef'].values[0]
                sig  = sub['sig'].values[0]
                pval = sub['p_value'].values[0]
                row += f"  {coef:>8.3f}{sig:<3} ({pval:.2f})"
        output_text.append(row)

    # Cumulative + R2 summary
    output_text.append(f"\n  SUMMARY:")
    output_text.append(
        f"  {'Group':<25} {'Donors':>7} {'N obs':>7} "
        f"{'Infra Cumul':>13} {'Battle Cumul':>13}"
    )
    output_text.append(f"  {'-'*68}")
    for donor_group in DONOR_GROUPS:
        sub = results_df[
            (results_df['donor_group'] == donor_group) &
            (results_df['aid_type']    == aid_type)
        ]
        if len(sub) == 0:
            continue
        n_donors   = sub['n_donors'].values[0]
        infra_cum  = sub[sub['variable'].isin(
            [f'infra_lag{l}' for l in range(1, N_LAGS+1)])]['coef'].sum()
        battle_cum = sub[sub['variable'].isin(
            [f'battle_lag{l}' for l in range(1, N_LAGS+1)])]['coef'].sum()
        output_text.append(
            f"  {GROUP_LABELS[donor_group]:<25} {n_donors:>7} "
            f"{'':>7} {infra_cum:>13.3f} {battle_cum:>13.3f}"
        )

for line in output_text:
    print(line)

# ── 5. SAVE ───────────────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

full_output = '\n'.join(output_text) + '\n\n' + '\n'.join(summary_lines)
with open(OUTPUT_TXT, 'w') as f:
    f.write(full_output)
print(f"Saved: {OUTPUT_TXT}")
print("\nDone.")