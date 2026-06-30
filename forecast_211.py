"""
211 Call Volume Forecast Pipeline
Usage:
    python forecast_211.py --refit      # weekly: read CSV, fit model, save pickle, write forecast
    python forecast_211.py --forecast   # daily:  load pickle, generate fresh forecast, write CSV
"""

import argparse
import logging
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from prophet import Prophet

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR = Path(r'C:\Users\powerbi\211_forecast')
MODEL_PATH = WORK_DIR / 'prophet_model.pkl'
LOG_PATH = WORK_DIR / 'forecast.log'

# Source data — 211FactCalls.csv exported from Power BI (refreshed manually or via Power Automate).
# Update this path if the file moves.
INPUT_CSV = Path(r'C:\Users\powerbi\211 Call Volume Forecasting\211FactCalls.csv')

# Local path where OneDrive has synced the SharePoint folder on UW-D10.
OUTPUT_DIR = Path(r'C:\Users\powerbi\UNITED WAY OF CENTRAL AND SOUTHERN UTAH\Data & Impact - PowerBI\Data CSVs\211 Forecasting')

FORECAST_CSV = OUTPUT_DIR / '211FactForecast.csv'

# ── Data quality ───────────────────────────────────────────────────────────────
BAD_DATES = {
    d.date() for d in pd.to_datetime([
        # Dec 2022: logging failure — calls bulk-dumped onto Dec 13
        '2022-12-01', '2022-12-02', '2022-12-03', '2022-12-04', '2022-12-05',
        '2022-12-06', '2022-12-07', '2022-12-08', '2022-12-09', '2022-12-10',
        '2022-12-11', '2022-12-12', '2022-12-13', '2022-11-30',
        # Other confirmed data errors
        '2021-02-01', '2021-02-02', '2021-02-03', '2021-02-08', '2021-02-09',
        '2021-01-25', '2021-01-29', '2020-03-20',
    ])
}

# ── Model config ───────────────────────────────────────────────────────────────
REGRESSORS = [
    'is_covid', 'is_holiday', 'is_vita', 'is_pioneer_day',
    'is_tax_deadline', 'is_day_before_holiday', 'is_day_after_holiday',
]


# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    fmt = '%(asctime)s | %(levelname)s | %(message)s'
    logging.basicConfig(filename=LOG_PATH, level=logging.INFO, format=fmt, datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# ── Data loading ───────────────────────────────────────────────────────────────
def load_csv() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f'Source file not found: {INPUT_CSV}\n'
            'Export 211FactCalls from Power BI and save it to that path (see README).'
        )
    df = pd.read_csv(
        INPUT_CSV,
        usecols=['InteractionDate'],
        parse_dates=['InteractionDate'],
        low_memory=False,
    )
    logging.info(f'Loaded {len(df):,} records from {INPUT_CSV.name}')
    return df


# ── Transformation ─────────────────────────────────────────────────────────────
def transform(df_raw: pd.DataFrame) -> pd.DataFrame:
    # InteractionDate is already in Mountain Time (Power BI M code converts it) — no tz conversion needed
    df_hourly = (
        df_raw
        .groupby(df_raw['InteractionDate'].dt.floor('h'))
        .size()
        .reset_index(name='y')
        .rename(columns={'InteractionDate': 'ds'})
    )

    # Zero-fill gaps so Prophet sees a continuous series
    full_range = pd.date_range(df_hourly['ds'].min(), df_hourly['ds'].max(), freq='h')
    df_hourly = df_hourly.set_index('ds').reindex(full_range, fill_value=0).reset_index()
    df_hourly.columns = ['ds', 'y']

    # Drop confirmed bad dates
    df_hourly = df_hourly[~df_hourly['ds'].dt.date.isin(BAD_DATES)].reset_index(drop=True)

    # Keep only operating hours (8 AM – 4 PM slots, 9 per day)
    df_hourly = df_hourly[df_hourly['ds'].dt.hour.between(8, 16)].copy()

    logging.info(f'After transform: {len(df_hourly):,} rows (operating hours, bad dates removed)')
    return df_hourly


# ── Regressors ─────────────────────────────────────────────────────────────────
def add_regressors(df: pd.DataFrame) -> pd.DataFrame:
    all_years = df['ds'].dt.year.unique().tolist()
    us_hols = holidays.US(years=all_years)
    holiday_dates = set(us_hols.keys())
    pioneer_dates = {date(yr, 7, 24) for yr in all_years}
    all_special = holiday_dates | pioneer_dates

    df = df.copy()
    df['is_covid'] = (
        (df['ds'] >= '2020-03-01') & (df['ds'] < '2021-01-01')
    ).astype(int)
    df['is_holiday'] = df['ds'].dt.date.isin(holiday_dates).astype(int)
    df['is_vita'] = (
        (df['ds'].dt.month <= 3) |
        ((df['ds'].dt.month == 4) & (df['ds'].dt.day <= 15))
    ).astype(int)
    df['is_pioneer_day'] = (
        (df['ds'].dt.month == 7) & (df['ds'].dt.day == 24)
    ).astype(int)
    df['is_tax_deadline'] = (
        (df['ds'].dt.month == 4) & (df['ds'].dt.day == 15)
    ).astype(int)
    df['is_day_before_holiday'] = df['ds'].dt.date.isin(
        {d - timedelta(days=1) for d in all_special}
    ).astype(int)
    df['is_day_after_holiday'] = df['ds'].dt.date.isin(
        {d + timedelta(days=1) for d in all_special}
    ).astype(int)
    return df


# ── Model ──────────────────────────────────────────────────────────────────────
def fit_model(df: pd.DataFrame) -> Prophet:
    df_prophet = add_regressors(df[['ds', 'y']])

    m = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.15,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    for reg in REGRESSORS:
        m.add_regressor(reg)

    logging.info(f'Fitting Prophet on {len(df_prophet):,} rows...')
    m.fit(df_prophet)
    logging.info('Model fit complete')
    return m


def save_model(m: Prophet) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(m, f)
    logging.info(f'Model saved → {MODEL_PATH}')


def load_model() -> Prophet:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'No model found at {MODEL_PATH}. Run --refit before --forecast.'
        )
    with open(MODEL_PATH, 'rb') as f:
        m = pickle.load(f)
    logging.info(f'Model loaded ← {MODEL_PATH}')
    return m


# ── Forecast ───────────────────────────────────────────────────────────────────
def generate_forecast(m: Prophet) -> pd.DataFrame:
    today = pd.Timestamp('today').normalize()
    future_hours = pd.DatetimeIndex([
        today + pd.Timedelta(days=d, hours=h)
        for d in range(91)
        for h in range(8, 17)
    ])
    future = add_regressors(pd.DataFrame({'ds': future_hours}))
    fc = m.predict(future)
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        fc[col] = fc[col].clip(lower=0)
    logging.info(f'Forecast generated: {len(fc):,} rows (91 days, operating hours)')
    return fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# ── Output ─────────────────────────────────────────────────────────────────────
def write_forecast(fc: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fc.to_csv(FORECAST_CSV, index=False)
    logging.info(f'Forecast written: {len(fc):,} rows → {FORECAST_CSV}')


# ── Jobs ───────────────────────────────────────────────────────────────────────
def run_refit() -> None:
    logging.info('=== REFIT JOB START ===')
    df_raw = load_csv()
    df = transform(df_raw)
    m = fit_model(df)
    save_model(m)
    fc = generate_forecast(m)
    write_forecast(fc)
    logging.info('=== REFIT JOB COMPLETE ===')


def run_forecast() -> None:
    logging.info('=== FORECAST JOB START ===')
    m = load_model()
    fc = generate_forecast(m)
    write_forecast(fc)
    logging.info('=== FORECAST JOB COMPLETE ===')


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    setup_logging()
    parser = argparse.ArgumentParser(description='211 Call Volume Forecast Pipeline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--refit', action='store_true',
                       help='Read CSV, refit Prophet, save pickle, write forecast (run weekly)')
    group.add_argument('--forecast', action='store_true',
                       help='Load pickle, generate 91-day forecast, write 211FactForecast.csv (run daily)')
    args = parser.parse_args()

    try:
        if args.refit:
            run_refit()
        else:
            run_forecast()
    except Exception as e:
        logging.exception(f'Job failed: {e}')
        sys.exit(1)
