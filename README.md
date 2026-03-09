# Conflict Escalation and Donor Response in Ukraine

Authors: Benita Besa & Mingyuan Song | Supervisor: Professor Charles Becker

## Research Question
Do infrastructure attacks versus battlefield events predict differential
humanitarian versus military aid responses across donor types?

## Data
- **Kiel Ukraine Support Tracker** (Release 27) — bilateral aid allocations
  by donor, month, and aid type in EUR millions
- **ACLED** — weekly conflict event counts aggregated to monthly,
  covering infrastructure attacks and battlefield events (Feb 2022 – Jan 2025)

## How to Run
```bash
# 1. Activate environment and install dependencies
python -m venv venv && source venv/bin/activate
pip install pandas numpy matplotlib statsmodels linearmodels numbers-parser

# 2. Clean and merge data (run from scripts/)
python kiel_ukraine_support_finder_data_cleaning_script.py
python merge_panel.py

# 3. Run regressions
python regression_baseline.py
python regression_lag_sensitivity.py
python regression_donor_heterogeneity.py
python regression_aid_size_split.py

```
## Methods
Distributed lag panel regression with donor and month fixed effects,
standard errors clustered at the donor level. Event study around 9
infrastructure spike months. Robustness checks at 4, 6, and 9 lag windows.
