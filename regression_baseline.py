"""
Baseline Distributed Lag Panel Regression
==========================================
Model: Aid_{i,t,k} = α + Σ β_l·Conflict_{t-l} + Σ γ_l·Infra_{t-l} + δ_i + λ_t + ε
  - Estimated separately for each aid type k (Military, Financial, Humanitarian)
  - 6 lags for both treatment variables
  - Donor fixed effects (δ_i) via EntityEffects
  - Month fixed effects (λ_t) via month dummies
  - Standard errors clustered at donor level

Input:  acled_monthly_clean.csv + kiel_panel_clean.csv
Output: regression_results.csv  (all coefficients, tidy format)
        regression_summary.txt  (full readable model output)

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
AID_TYPES  = ['Military', 'Financial', 'Humanitarian']
OUTPUT_CSV = '../results/regression_results.csv'
OUTPUT_TXT = '../results/regression_summary.txt'

# ── 1. LOAD AND BUILD MERGED PANEL ────────────────────────────────────────────
print("Loading data ...")
acled = pd.read_csv('../data/acled_monthly_clean.csv')
kiel  = pd.read_csv('../data/kiel_panel_clean.csv')

acled = acled[acled['year_month'] >= '2022-02'].copy()
acled_merge = acled.drop(columns=['t'])
panel = kiel.merge(acled_merge, on='year_month', how='left')

panel = panel.sort_values(['donor', 'aid_type_general', 'year_month']).reset_index(drop=True)

for lag in range(1, N_LAGS + 1):
    panel[f'infra_lag{lag}']  = panel.groupby(['donor', 'aid_type_general'])['infrastructure_attacks'].shift(lag)
    panel[f'battle_lag{lag}'] = panel.groupby(['donor', 'aid_type_general'])['battlefield_events'].shift(lag)

print(f"  Panel shape: {panel.shape}")
print(f"  Period: {panel['year_month'].min()} to {panel['year_month'].max()}")

# ── 2. REGRESSION FUNCTION ────────────────────────────────────────────────────
def run_regression(df, aid_type, n_lags):
    lag_cols = ([f'infra_lag{l}'  for l in range(1, n_lags+1)] +
                [f'battle_lag{l}' for l in range(1, n_lags+1)])

    sub = df[df['aid_type_general'] == aid_type].dropna(subset=lag_cols).copy()
    n_dropped = len(df[df['aid_type_general'] == aid_type]) - len(sub)
    print(f"\n  [{aid_type}] Usable rows: {len(sub)} (dropped {n_dropped} for lag NaNs)")

    sub['date'] = pd.to_datetime(sub['year_month'])
    sub = sub.set_index(['donor', 'date'])

    month_dummies = pd.get_dummies(sub['month_num'], prefix='month', drop_first=True).astype(float)
    sub = pd.concat([sub, month_dummies], axis=1)

    infra_terms  = ' + '.join([f'infra_lag{l}'  for l in range(1, n_lags+1)])
    battle_terms = ' + '.join([f'battle_lag{l}' for l in range(1, n_lags+1)])
    month_terms  = ' + '.join(month_dummies.columns.tolist())
    formula = f'aid_eur_m ~ {infra_terms} + {battle_terms} + {month_terms} + EntityEffects'

    model  = PanelOLS.from_formula(formula, data=sub, drop_absorbed=True)
    result = model.fit(cov_type='clustered', cluster_entity=True)

    tidy = pd.DataFrame({
        'aid_type': aid_type,
        'variable': result.params.index,
        'coef':     result.params.values,
        'std_err':  result.std_errors.values,
        't_stat':   result.tstats.values,
        'p_value':  result.pvalues.values,
        'ci_lower': result.params.values - 1.96 * result.std_errors.values,
        'ci_upper': result.params.values + 1.96 * result.std_errors.values,
    })
    tidy['sig'] = tidy['p_value'].apply(
        lambda p: '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.10 else ''))
    )
    return result, tidy

# ── 3. RUN FOR EACH AID TYPE ──────────────────────────────────────────────────
print("\nRunning regressions ...")
all_results   = []
fitted_models = {}
summary_lines = []

for aid_type in AID_TYPES:
    result, tidy = run_regression(panel, aid_type, N_LAGS)
    all_results.append(tidy)
    fitted_models[aid_type] = result

    key_vars = ([f'infra_lag{l}'  for l in range(1, N_LAGS+1)] +
                [f'battle_lag{l}' for l in range(1, N_LAGS+1)])
    key = tidy[tidy['variable'].isin(key_vars)]

    print(f"\n  -- {aid_type.upper()} AID --")
    print(f"  {'Variable':<15} {'Coef':>9} {'Std Err':>9} {'p-val':>8} {'':>5}")
    print(f"  {'-'*50}")
    for _, row in key.iterrows():
        print(f"  {row['variable']:<15} {row['coef']:>9.3f} {row['std_err']:>9.3f} {row['p_value']:>8.3f} {row['sig']:>5}")

    infra_cum  = tidy[tidy['variable'].isin([f'infra_lag{l}'  for l in range(1, N_LAGS+1)])]['coef'].sum()
    battle_cum = tidy[tidy['variable'].isin([f'battle_lag{l}' for l in range(1, N_LAGS+1)])]['coef'].sum()
    print(f"\n  R2 (within):              {result.rsquared:.4f}")
    print(f"  Observations:             {result.nobs}")
    print(f"  Cumulative infra effect:  {infra_cum:.3f}")
    print(f"  Cumulative battle effect: {battle_cum:.3f}")

    summary_lines.append(f"\n{'='*60}\n{aid_type.upper()} AID\n{'='*60}")
    summary_lines.append(str(result.summary))

# ── 4. COMPARISON TABLES ──────────────────────────────────────────────────────
results_df = pd.concat(all_results, ignore_index=True)

print(f"\n\n-- INFRA LAG COEFFICIENTS BY AID TYPE --")
print("(+) = more infrastructure attacks -> more aid that many months later\n")
infra_vars = [f'infra_lag{l}' for l in range(1, N_LAGS+1)]
comp_infra = results_df[results_df['variable'].isin(infra_vars)].pivot(
    index='variable', columns='aid_type', values='coef')[AID_TYPES]
comp_infra.index = [f'lag {i+1}' for i in range(N_LAGS)]
print(comp_infra.round(3).to_string())

print(f"\n-- BATTLE LAG COEFFICIENTS BY AID TYPE --\n")
battle_vars = [f'battle_lag{l}' for l in range(1, N_LAGS+1)]
comp_battle = results_df[results_df['variable'].isin(battle_vars)].pivot(
    index='variable', columns='aid_type', values='coef')[AID_TYPES]
comp_battle.index = [f'lag {i+1}' for i in range(N_LAGS)]
print(comp_battle.round(3).to_string())

print(f"\n-- P-VALUES: INFRA LAGS BY AID TYPE --\n")
comp_pval = results_df[results_df['variable'].isin(infra_vars)].pivot(
    index='variable', columns='aid_type', values='p_value')[AID_TYPES]
comp_pval.index = [f'lag {i+1}' for i in range(N_LAGS)]
print(comp_pval.round(3).to_string())

# ── 5. SAVE ───────────────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

with open(OUTPUT_TXT, 'w') as f:
    f.write('\n'.join(summary_lines))
print(f"Saved: {OUTPUT_TXT}")
print("\nDone.")