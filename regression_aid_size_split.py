"""
Aid Size Split Analysis — Military Aid
=======================================
Addresses Professor Becker's concern that large military packages (requiring
parliamentary authorization and procurement pipelines) may have longer lag
structures than small packages (which can be disbursed more quickly).

Split strategy:
  - Compute each donor's average monthly military aid over the full panel
  - Split donors at the median into "large" and "small" package groups
  - Re-estimate the 6-lag distributed lag model separately for each group
  - Compare peak lag (lag with largest coefficient) between groups

Note: We split on donor-level averages rather than row-level values because
the panel is 70% zeros — a row-level median split would be degenerate.

Input:  acled_monthly_clean.csv + kiel_panel_clean.csv
Output: aid_size_split_results.csv
        aid_size_split_summary.txt

Author: Benita Besa, Mingyuan Song
Course: Econ 590, Duke University
"""

import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS
import warnings
warnings.filterwarnings('ignore')

# ── 0. CONFIG ─────────────────────────────────────────────────────────────────
N_LAGS     = 6
OUTPUT_CSV = '../results/aid_size_split_results.csv'
OUTPUT_TXT = '../results/aid_size_split_summary.txt'

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

# ── 2. CREATE DONOR SIZE GROUPS ───────────────────────────────────────────────
mil_panel = panel[panel['aid_type_general'] == 'Military'].copy()

# Average monthly military commitment per donor
donor_avg = (mil_panel.groupby('donor')['aid_eur_m']
             .mean()
             .reset_index()
             .rename(columns={'aid_eur_m': 'avg_monthly_mil'}))

median_avg = donor_avg['avg_monthly_mil'].median()
donor_avg['size_group'] = donor_avg['avg_monthly_mil'].apply(
    lambda x: 'large' if x >= median_avg else 'small'
)

print(f"\n  Median donor avg monthly military: {median_avg:.3f} EUR m")
print(f"\n  Large donors ({(donor_avg['size_group']=='large').sum()}):")
for _, row in donor_avg[donor_avg['size_group']=='large'].sort_values(
        'avg_monthly_mil', ascending=False).iterrows():
    print(f"    {row['donor']:<20} avg={row['avg_monthly_mil']:.1f} EUR m")

print(f"\n  Small donors ({(donor_avg['size_group']=='small').sum()}):")
for _, row in donor_avg[donor_avg['size_group']=='small'].sort_values(
        'avg_monthly_mil', ascending=False).iterrows():
    print(f"    {row['donor']:<20} avg={row['avg_monthly_mil']:.1f} EUR m")

# Merge size group back onto military panel
mil_panel = mil_panel.merge(donor_avg[['donor', 'size_group']], on='donor', how='left')

# ── 3. REGRESSION FUNCTION ────────────────────────────────────────────────────
def run_regression(df, size_group, n_lags):
    lag_cols = ([f'infra_lag{l}'  for l in range(1, n_lags+1)] +
                [f'battle_lag{l}' for l in range(1, n_lags+1)])

    sub = df[df['size_group'] == size_group].dropna(subset=lag_cols).copy()
    n_donors = sub['donor'].nunique()

    print(f"\n  [{size_group.upper()} packages] Donors: {n_donors}, Rows: {len(sub)}")

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
        'size_group': size_group,
        'n_donors':   n_donors,
        'variable':   result.params.index,
        'coef':       result.params.values,
        'std_err':    result.std_errors.values,
        'p_value':    result.pvalues.values,
        'ci_lower':   result.params.values - 1.96 * result.std_errors.values,
        'ci_upper':   result.params.values + 1.96 * result.std_errors.values,
    })
    tidy['sig'] = tidy['p_value'].apply(
        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    )
    return result, tidy

# ── 4. RUN REGRESSIONS ────────────────────────────────────────────────────────
print("\nRunning aid size split regressions ...")
all_results   = []
summary_lines = []

for size_group in ['large', 'small']:
    result, tidy = run_regression(mil_panel, size_group, N_LAGS)
    all_results.append(tidy)
    summary_lines.append(
        f"\n{'='*60}\n{size_group.upper()} PACKAGE DONORS — MILITARY AID\n{'='*60}"
    )
    summary_lines.append(str(result.summary))

results_df = pd.concat(all_results, ignore_index=True)

# ── 5. COMPARISON TABLE ───────────────────────────────────────────────────────
output_text = []
output_text.append("AID SIZE SPLIT — MILITARY AID ONLY")
output_text.append("Split: donor avg monthly military aid above/below median")
output_text.append(f"Median threshold: {median_avg:.3f} EUR m/month")
output_text.append("Significance: *** p<0.01  ** p<0.05  * p<0.10")

output_text.append(f"\n{'='*65}")
output_text.append("COEFFICIENT COMPARISON: LARGE vs SMALL PACKAGE DONORS")
output_text.append(f"{'='*65}")
output_text.append(
    f"\n  {'Variable':<15} {'Large packages':>20} {'Small packages':>20}"
)
output_text.append(f"  {'-'*57}")

key_vars = ([f'infra_lag{l}'  for l in range(1, N_LAGS+1)] +
            [f'battle_lag{l}' for l in range(1, N_LAGS+1)])

for var in key_vars:
    row = f"  {var:<15}"
    for size_group in ['large', 'small']:
        sub = results_df[
            (results_df['size_group'] == size_group) &
            (results_df['variable']   == var)
        ]
        if len(sub) == 0:
            row += f"  {'n/a':>18}"
        else:
            coef = sub['coef'].values[0]
            sig  = sub['sig'].values[0]
            pval = sub['p_value'].values[0]
            row += f"  {coef:>10.3f}{sig:<3} ({pval:.2f})"
    output_text.append(row)

# Peak lag analysis
output_text.append(f"\n{'='*65}")
output_text.append("PEAK LAG ANALYSIS")
output_text.append(f"{'='*65}")
output_text.append("(Which lag has the largest absolute coefficient?)\n")

for var_type, label in [('infra', 'Infrastructure attacks'), ('battle', 'Battlefield events')]:
    output_text.append(f"  {label}:")
    for size_group in ['large', 'small']:
        lag_coefs = []
        for lag in range(1, N_LAGS+1):
            sub = results_df[
                (results_df['size_group'] == size_group) &
                (results_df['variable']   == f'{var_type}_lag{lag}')
            ]
            if len(sub) > 0:
                lag_coefs.append((lag, sub['coef'].values[0], sub['p_value'].values[0]))

        if lag_coefs:
            peak_lag, peak_coef, peak_p = max(lag_coefs, key=lambda x: abs(x[1]))
            sig = '***' if peak_p < 0.01 else '**' if peak_p < 0.05 else '*' if peak_p < 0.10 else ''
            cum = sum(c for _, c, _ in lag_coefs)
            output_text.append(
                f"    {size_group.upper():<8}: peak at lag {peak_lag} "
                f"(coef={peak_coef:.3f}{sig}, p={peak_p:.3f})  "
                f"cumulative={cum:.3f}"
            )
    output_text.append("")

# Summary table
output_text.append(f"{'='*65}")
output_text.append("SUMMARY")
output_text.append(f"{'='*65}")
output_text.append(
    f"\n  {'Group':<18} {'Donors':>8} {'N obs':>8} "
    f"{'Infra Cumul':>13} {'Battle Cumul':>13} {'R2':>8}"
)
output_text.append(f"  {'-'*70}")

for size_group, result_obj in zip(['large', 'small'],
                                   [r for r in [None, None]]):
    sub = results_df[results_df['size_group'] == size_group]
    n_donors   = sub['n_donors'].values[0]
    infra_cum  = sub[sub['variable'].isin(
        [f'infra_lag{l}'  for l in range(1, N_LAGS+1)])]['coef'].sum()
    battle_cum = sub[sub['variable'].isin(
        [f'battle_lag{l}' for l in range(1, N_LAGS+1)])]['coef'].sum()
    output_text.append(
        f"  {size_group.upper()+' packages':<18} {n_donors:>8} "
        f"{'':>8} {infra_cum:>13.3f} {battle_cum:>13.3f}"
    )

for line in output_text:
    print(line)

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

full_output = '\n'.join(output_text) + '\n\n' + '\n'.join(summary_lines)
with open(OUTPUT_TXT, 'w') as f:
    f.write(full_output)
print(f"Saved: {OUTPUT_TXT}")
print("\nDone.")