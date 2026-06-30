# 211 Call Volume Forecasting Pipeline — Build Spec

## Goal

Build a Python pipeline that reads 211 call data from a CSV export, generates an hourly forecast of call volume for the next 91 days, and writes the forecast to a SharePoint-synced folder where a standalone Power BI report will consume it.

The final deliverable for end users is a Power BI dashboard. This spec covers the data pipeline that feeds it. The .pbix build is a separate step done in Power BI Desktop, not here.

## Architecture

```
211FactCalls.csv  (exported from Power BI, manually refreshed or via Power Automate)
    ↓
Python script on UW-D10  (forecast_211.py)
    ↓
211FactForecast.csv  in OneDrive-synced SharePoint folder
    ↓
Power BI Desktop file (separate semantic model, NOT the USM)
    — imports 211FactCalls natively from Salesforce (existing connection)
    — imports 211FactForecast.csv from SharePoint folder
    ↓
Published to Data & Impact PBI Service workspace
    ↓
Dashboard
```

This is a **standalone semantic model**, completely separate from the Unified Semantic Model (USM). Power BI handles actuals natively via its existing Salesforce connection; this pipeline only produces the forecast CSV.

**Why not pull Salesforce directly?** The `powerbi` Salesforce user returned 0 Case records via the REST API (likely an org mismatch or sharing-visibility issue). Rather than debug Salesforce admin permissions, the pipeline reads from the CSV that Power BI already exports — bypassing the Salesforce API entirely.

**Why not use the Power BI REST API?** Execute Queries (DAX) via service principal requires Azure AD app registration, a tenant admin setting, and likely Premium/PPU capacity. United Way is on Power BI Pro with no confirmed Azure AD admin access. CSV bridge is simpler and has no new IT dependencies.

## Runtime environment

- **Host:** UW-D10 (always powered on, but user may be logged out)
- **Account:** `powerbi` Windows account (tied to this machine, password does not rotate)
- **Scheduler:** Windows Task Scheduler, configured "Run whether user is logged on or not"
- **Python:** `C:\Users\powerbi\AppData\Local\anaconda3\python.exe`

## Schedule

Two scheduled tasks:

1. **Weekly model refit** — Sunday at 9:00 PM MT. Reads the CSV, refits Prophet from scratch, pickles the model to disk, writes an updated forecast.
2. **Daily forecast generation** — Every morning at 7:00 AM MT. Loads the pickled model, generates a fresh 91-day forecast, writes `211FactForecast.csv`. Does NOT read the source CSV — only the model is needed.

The daily job does NOT refit. Day-to-day forecast changes are attributable purely to the passage of time (new future timestamps), not model drift.

## Data source

**`211FactCalls.csv`** — located at:
```
C:\Users\powerbi\211 Call Volume Forecasting\211FactCalls.csv
```

This file is currently refreshed manually: someone exports the `211FactCalls` table from Power BI and overwrites this file. A future Power Automate flow could automate this export on a weekly schedule, dropping the CSV into the SharePoint folder where `INPUT_CSV` points.

If the file location changes, update `INPUT_CSV` near the top of `forecast_211.py`.

### Source CSV columns

The `211FactCalls.csv` file (as currently exported from Power BI) has these columns:

```
InteractionDate, CallType, Status, State, InteractionHourKey
```

Only `InteractionDate` is used by the pipeline. `InteractionDate` is already in **Mountain Time** (Power BI's M code converts it from Salesforce's UTC `CreatedDate`) — no timezone conversion is needed in Python.

### Filters

**No call-type or status filters.** Include all calls — all `CallType` values, all `Status` values including Disconnected. This matches the Streamlit prototype and the `211FactCalls.csv` training data.

### Aggregation

Truncate `InteractionDate` to the hour and count records to get hourly call volume:

```
ds (datetime, hour-truncated, MT, tz-naive) | y (int, call count)
```

Zero-fill any hours with no calls (reindex to a continuous hourly range). Filter to operating hours 8–16 (9 slots/day). Drop known bad dates (see `BAD_DATES` in `forecast_211.py`).

## Prophet model

```python
REGRESSORS = [
    'is_covid', 'is_holiday', 'is_vita', 'is_pioneer_day',
    'is_tax_deadline', 'is_day_before_holiday', 'is_day_after_holiday',
]

m = Prophet(
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.15,
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
)
for reg in REGRESSORS:
    m.add_regressor(reg)
```

See CLAUDE.md for the definition of each regressor. All 7 must be present in both the training frame and any future dataframe passed to `predict()`.

After fitting, pickle to disk:

```python
import pickle
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
```

## Forecast generation

Build 91 days of future operating-hours timestamps (hours 8–16, 9 slots/day), add all 7 regressors, then predict:

```python
today = pd.Timestamp('today').normalize()
future_hours = pd.DatetimeIndex([
    today + pd.Timedelta(days=d, hours=h)
    for d in range(91)
    for h in range(8, 17)
])
future = add_regressors(pd.DataFrame({'ds': future_hours}))
forecast = model.predict(future)
```

Output columns: `ds`, `yhat`, `yhat_lower`, `yhat_upper`.

## Output file

### `211FactForecast.csv`

Written to the OneDrive-synced SharePoint folder:
```
C:\Users\powerbi\UNITED WAY OF CENTRAL AND SOUTHERN UTAH\Data & Impact - PowerBI\Data CSVs\211 Forecasting\211FactForecast.csv
```

Next 91 days of hourly forecast (operating hours only). Overwritten each run.

```
ds, yhat, yhat_lower, yhat_upper
```

Columns: `ds` (ISO datetime, hour grain, MT, tz-naive), forecast point + 80% CI bounds (Prophet default). Values clipped at 0 (calls can't be negative).

No separate actuals CSV — Power BI already has actuals natively via its Salesforce connection (`211FactCalls`). Writing a duplicate would serve no purpose.

## Caching

- Pickled model: `C:\Users\powerbi\211_forecast\prophet_model.pkl`
- Log file: `C:\Users\powerbi\211_forecast\forecast.log`

This working directory is separate from the SharePoint output folder and is created automatically on first run.

## Logging

Append-only log file. Each run writes: timestamp, job type (refit vs forecast), row counts, success/failure.

No email-on-failure for v1. If runs go silent, the Power BI report's "last updated" timestamp will surface the problem.

## What this script does NOT do

- No Salesforce API calls
- No forecast history retention (each run overwrites)
- No accuracy metrics (MAE/MAPE) — out of scope for v1
- No staffing recommendations — explicitly out of scope per Emily/Bill
- No email alerts
- No writing to USM
- No DirectQuery, no XMLA, no Power BI API calls — output is just a CSV

## Deliverables

1. `forecast_211.py` — single script with `--refit` / `--forecast` CLI modes
2. `requirements_pipeline.txt` — dependencies: `prophet`, `pandas`, `holidays`, `numpy`
3. `README.md` — setup instructions: install deps, refresh the CSV, confirm output path, Task Scheduler steps, first-run sanity check

## Open items

- Whether the `powerbi` Windows account has rights to create scheduled tasks that run when logged off (should be fine but verify)
- Future: Power Automate flow to automate the `211FactCalls.csv` refresh so the weekly refit always trains on current data without manual exports
