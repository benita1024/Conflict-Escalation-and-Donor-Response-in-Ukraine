"""
Lag Sensitivity Robustness Check
=================================
Re-estimates the baseline distributed lag panel regression at 3 lag windows:
  - 4 lags  (uses month dummies for time FE)
  - 6 lags  (baseline, uses month dummies for time FE)
  - 9 lags  (uses quarter dummies for time FE — fewer time periods available)

Note on 9-lag model: with only 27 usable months after dropping NaN lags,
using 11 month dummies alongside 18 lag variables causes rank deficiency.
We switch to quarter dummies (3 dummies) for the 9-lag window only.
This is noted in the robustness section.

Input:  acled_monthly_clean.csv + kiel_panel_clean.csv
Output: lag_sensitivity_results.csv
        lag_sensitivity_summary.txt

Author: Benita Besa, Mingyuan Song
Course: Econ 590, Duke University
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────
LAG_WINDOWS = [4, 6, 9]
AID_TYPES   = ['Military', 'Financial', 'Humanitarian']
OUTPUT_CSV  = '../results/lag_sensitivity_results.csv'
OUTPUT_TXT  = '../results/lag_sensitivity_summary.txt'

# ── 1. LOAD AND BUILD PANEL ───────────────────────────────────────────────────
print("Loading data ...")
acled = pd.read_csv('../data/acled_monthly_clean.csv')
kiel  = pd.read_csv('../data/kiel_panel_clean.csv')

acled = acled[acled['year_month'] >= '2022-02'].copy()
acled_merge = acled.drop(columns=['t'])
panel = kiel.merge(acled_merge, on='year_month', how='left')

panel = panel.sort_values(
    ['donor', 'aid_type_general', 'year_month']
).reset_index(drop=True)

# Pre-compute all lags up to 9
for lag in range(1, 10):
    panel[f'infra_lag{lag}']  = panel.groupby(
        ['donor', 'aid_type_general'])['infrastructure_attacks'].shift(lag)
    panel[f'battle_lag{lag}'] = panel.groupby(
        ['donor', 'aid_type_general'])['battlefield_events'].shift(lag)

# Add quarter variable for 9-lag time FE
panel['quarter'] = panel['month_num'].apply(lambda m: (m - 1) // 3 + 1)

print(f"  Panel shape: {panel.shape}")

# ── 2. REGRESSION FUNCTION ────────────────────────────────────────────────────
def run_regression(df, aid_type, n_lags):
    lag_cols = ([f'infra_lag{l}'  for l in range(1, n_lags+1)] +
                [f'battle_lag{l}' for l in range(1, n_lags+1)])

    sub = df[df['aid_type_general'] == aid_type].dropna(subset=lag_cols).copy()
    sub['date'] = pd.to_datetime(sub['year_month'])
    sub = sub.set_index(['donor', 'date'])

    # Use month dummies for 4 and 6 lag models, quarter dummies for 9
    if n_lags <= 6:
        time_dummies = pd.get_dummies(
            sub['month_num'], prefix='month', drop_first=True
        ).astype(float)
    else:
        time_dummies = pd.get_dummies(
            sub['quarter'], prefix='q', drop_first=True
        ).astype(float)

    sub = pd.concat([sub, time_dummies], axis=1)

    infra_terms  = ' + '.join([f'infra_lag{l}'  for l in range(1, n_lags+1)])
    battle_terms = ' + '.join([f'battle_lag{l}' for l in range(1, n_lags+1)])
    time_terms   = ' + '.join(time_dummies.columns.tolist())
    formula = (f'aid_eur_m ~ {infra_terms} + {battle_terms} '
               f'+ {time_terms} + EntityEffects')

    model  = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    tidy = pd.DataFrame({
        'n_lags':   n_lags,
        'aid_type': aid_type,
        'variable': result.params.index,
        'coef':     result.params.values,
        'std_err':  result.std_errors.values,
        'p_value':  result.pvalues.values,
        'ci_lower': result.params.values - 1.96 * result.std_errors.values,
        'ci_upper': result.params.values + 1.96 * result.std_errors.values,
    })
    tidy['sig'] = tidy['p_value'].apply(
        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    )
    return result, tidy

# ── 3. RUN ALL COMBINATIONS ───────────────────────────────────────────────────
print("\nRunning lag sensitivity checks ...")
all_results   = []
summary_lines = []

for n_lags in LAG_WINDOWS:
    print(f"\n  === LAG WINDOW: {n_lags} ===")
    for aid_type in AID_TYPES:
        result, tidy = run_regression(panel, aid_type, n_lags)
        all_results.append(tidy)

        infra_cum  = tidy[tidy['variable'].isin(
            [f'infra_lag{l}' for l in range(1, n_lags+1)])]['coef'].sum()
        battle_cum = tidy[tidy['variable'].isin(
            [f'battle_lag{l}' for l in range(1, n_lags+1)])]['coef'].sum()

        print(f"    [{aid_type}] N={result.nobs}  "
              f"R2={result.rsquared:.4f}  "
              f"Infra cumul={infra_cum:.3f}  "
              f"Battle cumul={battle_cum:.3f}")

        summary_lines.append(
            f"\n{'='*60}\nLAGS={n_lags} | {aid_type.upper()}\n{'='*60}"
        )
        summary_lines.append(str(result.summary))

results_df = pd.concat(all_results, ignore_index=True)

# ── 4. COMPARISON TABLES ──────────────────────────────────────────────────────
output_text = []
output_text.append("LAG SENSITIVITY ROBUSTNESS CHECK")
output_text.append("Coefficients shown as:  coef*** (p-val)")
output_text.append("Significance: *** p<0.01  ** p<0.05  * p<0.10")
output_text.append("Note: 9-lag model uses quarter dummies (vs month dummies for 4/6 lag)")

for aid_type in AID_TYPES:
    output_text.append(f"\n{'='*70}")
    output_text.append(f"AID TYPE: {aid_type.upper()}")
    output_text.append(f"{'='*70}")

    for var_type in ['infra', 'battle']:
        label = 'INFRASTRUCTURE' if var_type == 'infra' else 'BATTLEFIELD'
        output_text.append(f"\n{label} LAG COEFFICIENTS:")
        output_text.append(
            f"  {'Lag':<14} {'4 lags':>18} {'6 lags (base)':>18} {'9 lags':>18}"
        )
        output_text.append(f"  {'-'*70}")

        for lag in range(1, 10):
            row = f"  lag {lag:<10}"
            for n_lags in LAG_WINDOWS:
                sub = results_df[
                    (results_df['n_lags']   == n_lags) &
                    (results_df['aid_type'] == aid_type) &
                    (results_df['variable'] == f'{var_type}_lag{lag}')
                ]
                if len(sub) == 0:
                    row += f"  {'n/a':>16}"
                else:
                    coef = sub['coef'].values[0]
                    sig  = sub['sig'].values[0]
                    pval = sub['p_value'].values[0]
                    row += f"  {coef:>8.3f}{sig:<3} ({pval:.2f})"
            output_text.append(row)

    # Cumulative summary
    output_text.append(f"\nCUMULATIVE EFFECTS SUMMARY:")
    output_text.append(
        f"  {'Window':<10} {'Infra Cumul':>14} {'Battle Cumul':>14}"
    )
    output_text.append(f"  {'-'*40}")
    for n_lags in LAG_WINDOWS:
        sub = results_df[
            (results_df['n_lags']   == n_lags) &
            (results_df['aid_type'] == aid_type)
        ]
        infra_cum  = sub[sub['variable'].isin(
            [f'infra_lag{l}'  for l in range(1, n_lags+1)])]['coef'].sum()
        battle_cum = sub[sub['variable'].isin(
            [f'battle_lag{l}' for l in range(1, n_lags+1)])]['coef'].sum()
        output_text.append(
            f"  {n_lags} lags{'':<5} {infra_cum:>14.3f} {battle_cum:>14.3f}"
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