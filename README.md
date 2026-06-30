# 211 Call Volume Forecast — Pipeline Setup

This pipeline reads `211FactCalls.csv` (exported from Power BI), fits a Prophet model, and writes `211FactForecast.csv` to a SharePoint-synced folder for Power BI to consume as a forecast table.

---

## How it works

```
211FactCalls.csv  (exported from Power BI, placed in project folder)
    ↓
forecast_211.py  (reads CSV → fits Prophet → writes forecast)
    ↓
211FactForecast.csv  (written to SharePoint folder → consumed by Power BI report)
```

Two scheduled tasks run on UW-D10:
- **Weekly refit** (Sunday 9 PM): reads the CSV, refits Prophet from scratch, saves the model, writes a fresh forecast
- **Daily forecast** (7 AM daily): loads the saved model, generates a fresh 91-day forecast, writes updated CSV — no CSV read needed

---

## Files

| File | Purpose |
|---|---|
| `forecast_211.py` | Main pipeline — run with `--refit` or `--forecast` |
| `requirements_pipeline.txt` | Python dependencies for this pipeline |

---

## Step 1 — Install dependencies

Open Anaconda Prompt on UW-D10 and run:

```
pip install -r "C:\Users\powerbi\211 Call Volume Forecasting\requirements_pipeline.txt"
```

---

## Step 2 — Keep 211FactCalls.csv current

The pipeline reads `211FactCalls.csv` from:
```
C:\Users\powerbi\211 Call Volume Forecasting\211FactCalls.csv
```

This file needs to be refreshed periodically so the weekly model refit trains on current data. To export a fresh copy from Power BI:

1. Open Power BI Service and navigate to the dataset containing `211FactCalls`
2. Open the table in a report visual or use **Analyze in Excel**
3. Export the data to CSV and save it to the path above, overwriting the existing file

> **Future improvement:** A Power Automate flow can automate this export on a schedule, dropping a fresh CSV into the SharePoint folder automatically. That would make the pipeline fully hands-off. Out of scope for the current setup.

If the CSV location ever changes, update `INPUT_CSV` near the top of `forecast_211.py`.

---

## Step 3 — Confirm the SharePoint output path

The forecast CSV is written to:
```
C:\Users\powerbi\UNITED WAY OF CENTRAL AND SOUTHERN UTAH\Data & Impact - PowerBI\Data CSVs\211 Forecasting\211FactForecast.csv
```

This is already configured as `OUTPUT_DIR` in `forecast_211.py`. Confirm this folder exists and is synced by OneDrive before running for the first time.

---

## Step 4 — Python executable (already confirmed)

The Python executable on UW-D10 is:
```
C:\Users\powerbi\AppData\Local\anaconda3\python.exe
```

Note: running `python` alone in PowerShell will not work on this machine — always use the full path above.

---

## Step 5 — Create the weekly refit task

Runs every Sunday at 9:00 PM to refit the model on the latest CSV data.

1. Open **Task Scheduler** (search in Start menu)
2. Click **Create Task** (not "Create Basic Task")
3. **General tab:**
   - Name: `211 Forecast — Weekly Refit`
   - Select **Run whether user is logged on or not**
   - Check **Run with highest privileges**
4. **Triggers tab → New:**
   - Weekly, every Sunday, start time: 9:00 PM
5. **Actions tab → New:**
   - Program/script:
     ```
     C:\Users\powerbi\AppData\Local\anaconda3\python.exe
     ```
   - Add arguments:
     ```
     forecast_211.py --refit
     ```
   - Start in:
     ```
     C:\Users\powerbi\211 Call Volume Forecasting
     ```
6. Click **OK**, enter the `powerbi` account password when prompted.

---

## Step 6 — Create the daily forecast task

Runs every morning at 7:00 AM to write a fresh `211FactForecast.csv`.

1. In Task Scheduler, click **Create Task**
2. **General tab:**
   - Name: `211 Forecast — Daily Forecast`
   - Select **Run whether user is logged on or not**
   - Check **Run with highest privileges**
3. **Triggers tab → New:**
   - Daily, start time: 7:00 AM
4. **Actions tab → New:**
   - Program/script:
     ```
     C:\Users\powerbi\AppData\Local\anaconda3\python.exe
     ```
   - Add arguments:
     ```
     forecast_211.py --forecast
     ```
   - Start in:
     ```
     C:\Users\powerbi\211 Call Volume Forecasting
     ```
5. Click **OK**, enter the `powerbi` account password when prompted.

---

## Step 7 — First-run sanity check

Run both jobs manually first to verify everything works before relying on the schedule:

```
& C:\Users\powerbi\AppData\Local\anaconda3\python.exe "C:\Users\powerbi\211 Call Volume Forecasting\forecast_211.py" --refit
& C:\Users\powerbi\AppData\Local\anaconda3\python.exe "C:\Users\powerbi\211 Call Volume Forecasting\forecast_211.py" --forecast
```

Then verify:
1. `C:\Users\powerbi\211_forecast\forecast.log` — both jobs should show COMPLETE with no errors
2. The SharePoint-synced folder should contain `211FactForecast.csv`
3. Open `211FactForecast.csv` — should have 819 rows (91 days × 9 operating hours), dates starting today, `yhat` values in a reasonable range (~10–40 on weekdays, near 0 on weekends)

---

## Logs

Every run appends to:
```
C:\Users\powerbi\211_forecast\forecast.log
```

If the Power BI dashboard looks stale, check this file first to see whether jobs ran and succeeded.

---

## Working directory

Model pickle and log files are stored in:
```
C:\Users\powerbi\211_forecast\
```
This folder is created automatically on first run. It is separate from the SharePoint output folder.
