import warnings
warnings.filterwarnings('ignore')

from datetime import date, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from prophet import Prophet

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="211 Call Volume Forecast",
    page_icon="📞",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parent / '211FactCalls.csv'

BAD_DATES = {
    d.date() for d in pd.to_datetime([
        '2022-12-01', '2022-12-02', '2022-12-03', '2022-12-04', '2022-12-05',
        '2022-12-06', '2022-12-07', '2022-12-08', '2022-12-09', '2022-12-10',
        '2022-12-11', '2022-12-12', '2022-12-13', '2022-11-30',
        '2021-02-01', '2021-02-02', '2021-02-03', '2021-02-08', '2021-02-09',
        '2021-01-25', '2021-01-29', '2020-03-20',
    ])
}

REGRESSORS = [
    'is_covid', 'is_holiday', 'is_vita', 'is_pioneer_day',
    'is_tax_deadline', 'is_day_before_holiday', 'is_day_after_holiday',
]

HOURS = list(range(8, 17))  # operating hours: 8 AM through 4 PM

# ── Regressors ────────────────────────────────────────────────────────────────
def add_regressors(df: pd.DataFrame) -> pd.DataFrame:
    all_years     = df['ds'].dt.year.unique().tolist()
    us_hols       = holidays.US(years=all_years)
    holiday_dates = set(us_hols.keys())
    pioneer_dates = {date(yr, 7, 24) for yr in all_years}
    all_special   = holiday_dates | pioneer_dates

    df = df.copy()
    df['is_covid']   = (
        (df['ds'] >= '2020-03-01') & (df['ds'] < '2021-01-01')
    ).astype(int)
    df['is_holiday'] = df['ds'].dt.date.isin(holiday_dates).astype(int)
    df['is_vita']    = (
        (df['ds'].dt.month <= 3) |
        ((df['ds'].dt.month == 4) & (df['ds'].dt.day <= 15))
    ).astype(int)
    df['is_pioneer_day']  = (
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

# ── Pipeline ──────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA_PATH,
        usecols=['InteractionDate', 'CallType', 'Status', 'State'],
        parse_dates=['InteractionDate'],
        dtype={'CallType': 'category', 'Status': 'category', 'State': 'category'},
        low_memory=False,
    )
    df['InteractionDate'] = (
        pd.to_datetime(df['InteractionDate'], utc=True)
        .dt.tz_convert('US/Mountain')
        .dt.tz_localize(None)
    )
    return df


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_hourly = (
        df_raw
        .groupby(df_raw['InteractionDate'].dt.floor('h'))
        .size()
        .reset_index(name='y')
        .rename(columns={'InteractionDate': 'ds'})
    )
    full_range = pd.date_range(df_hourly['ds'].min(), df_hourly['ds'].max(), freq='h')
    df_hourly  = df_hourly.set_index('ds').reindex(full_range, fill_value=0).reset_index()
    df_hourly.columns = ['ds', 'y']
    df_hourly  = df_hourly[~df_hourly['ds'].dt.date.isin(BAD_DATES)].reset_index(drop=True)
    df_model   = df_hourly[df_hourly['ds'].dt.hour.between(8, 16)].copy()
    return add_regressors(df_model[['ds', 'y']])


def train_model(df_prophet: pd.DataFrame) -> Prophet:
    m = Prophet(
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.15,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    for reg in REGRESSORS:
        m.add_regressor(reg)
    m.fit(df_prophet)
    return m


def generate_forecast(m: Prophet) -> pd.DataFrame:
    today        = pd.Timestamp('today').normalize()
    future_hours = pd.DatetimeIndex([
        today + pd.Timedelta(days=d, hours=h)
        for d in range(91)
        for h in range(8, 17)
    ])
    future = add_regressors(pd.DataFrame({'ds': future_hours}))
    fc     = m.predict(future)
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        fc[col] = fc[col].clip(lower=0)
    return fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


@st.cache_resource
def run_pipeline():
    df_raw     = load_data()
    df_prophet = preprocess(df_raw)
    m          = train_model(df_prophet)
    forecast   = generate_forecast(m)
    return forecast, df_prophet[['ds', 'y']].copy(), len(df_raw), df_raw['InteractionDate'].max().date()


# ── Shared layout defaults ────────────────────────────────────────────────────
LAYOUT = dict(
    template='plotly_white',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    margin=dict(t=50, b=60, l=60, r=20),
    hovermode='x unified',
    yaxis=dict(gridcolor='rgba(200,200,200,0.3)', zeroline=False),
)

STEELBLUE  = 'steelblue'
BAND_COLOR = 'rgba(70,130,180,0.15)'
DARK       = '#333333'


def _band(x_vals, lower, upper, name=''):
    """Helper: shaded confidence band between lower and upper."""
    return go.Scatter(
        x=list(x_vals) + list(x_vals)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill='toself',
        fillcolor=BAND_COLOR,
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip',
        name=name,
    )


def _today_line(fig: go.Figure, x_val) -> None:
    """Add a dashed Today line — uses add_shape + add_annotation to avoid
    a Plotly bug in add_vline where annotation placement fails on date axes."""
    fig.add_shape(
        type='line',
        x0=x_val, x1=x_val,
        y0=0, y1=1,
        xref='x', yref='paper',
        line=dict(dash='dash', color='rgba(100,100,100,0.6)', width=1.5),
    )
    fig.add_annotation(
        x=x_val, y=1.04,
        xref='x', yref='paper',
        text='<b>Today</b>',
        showarrow=False,
        xanchor='center',
        font=dict(size=11, color='gray'),
    )


# ── Charts ────────────────────────────────────────────────────────────────────
def plot_3day_hourly(fc: pd.DataFrame, hist: pd.DataFrame) -> go.Figure:
    """2 days of actuals + 3 days of predictions, hour by hour."""
    today    = pd.Timestamp('today').normalize()
    all_days = [
        (today - pd.Timedelta(days=2)).date(),
        (today - pd.Timedelta(days=1)).date(),
        today.date(),
        (today + pd.Timedelta(days=1)).date(),
        (today + pd.Timedelta(days=2)).date(),
    ]
    N = len(HOURS)  # 9 slots per day

    # Map integer x-positions → hover label
    pos_info = {}
    for day_idx, d in enumerate(all_days):
        for h_idx, hour in enumerate(HOURS):
            h12    = hour % 12 or 12
            suffix = 'AM' if hour < 12 else 'PM'
            pos_info[day_idx * N + h_idx] = (
                pd.Timestamp(d).strftime('%A, %b %d'),
                f"{h12} {suffix}",
            )

    actual_x, actual_y = [], []
    pred_x, pred_y, pred_low, pred_high = [], [], [], []

    for day_idx, d in enumerate(all_days):
        is_actual = day_idx < 2
        for h_idx, hour in enumerate(HOURS):
            x_pos = day_idx * N + h_idx
            ts    = pd.Timestamp(d) + pd.Timedelta(hours=hour)
            if is_actual:
                row = hist[hist['ds'] == ts]
                if len(row) > 0:
                    actual_x.append(x_pos)
                    actual_y.append(float(row['y'].values[0]))
            else:
                row = fc[fc['ds'] == ts]
                if len(row) > 0:
                    pred_x.append(x_pos)
                    pred_y.append(float(row['yhat'].values[0]))
                    pred_low.append(float(row['yhat_lower'].values[0]))
                    pred_high.append(float(row['yhat_upper'].values[0]))

    fig = go.Figure()

    # Confidence band
    if pred_x:
        fig.add_trace(_band(pred_x, pred_low, pred_high))

    # Predicted line
    if pred_x:
        pred_cd = [[pos_info[x][0], pos_info[x][1]] for x in pred_x]
        fig.add_trace(go.Scatter(
            x=pred_x, y=pred_y,
            mode='lines+markers',
            name='Predicted',
            line=dict(color=STEELBLUE, width=2.5),
            marker=dict(size=7, color=STEELBLUE),
            customdata=pred_cd,
            hovertemplate='<b>%{customdata[0]}</b> · %{customdata[1]}<br>Predicted: <b>%{y:.0f}</b> calls<extra></extra>',
        ))

    # Actual line
    if actual_x:
        actual_cd = [[pos_info[x][0], pos_info[x][1]] for x in actual_x]
        fig.add_trace(go.Scatter(
            x=actual_x, y=actual_y,
            mode='lines+markers',
            name='Actual',
            line=dict(color=DARK, width=2.5),
            marker=dict(size=7, color=DARK),
            customdata=actual_cd,
            hovertemplate='<b>%{customdata[0]}</b> · %{customdata[1]}<br>Actual: <b>%{y:.0f}</b> calls<extra></extra>',
        ))

    # Day separator lines
    for i in range(1, 5):
        sep = i * N - 0.5
        clr = 'rgba(100,100,100,0.6)' if i == 2 else 'rgba(180,180,180,0.5)'
        fig.add_vline(x=sep, line_dash='dash', line_color=clr, line_width=1.2)

    # "Today" line
    _today_line(fig, 2 * N - 0.5)

    # Day name annotations along the top
    for day_idx, d in enumerate(all_days):
        label = 'Today' if day_idx == 2 else pd.Timestamp(d).strftime('%a %b %d')
        fig.add_annotation(
            x=day_idx * N + N // 2, y=1.01, xref='x', yref='paper',
            text=f'<b>{label}</b>', showarrow=False,
            font=dict(size=10, color='dimgray'),
        )

    # Hour tick labels (every other hour)
    tick_vals, tick_text = [], []
    for day_idx in range(5):
        for h_idx, hour in enumerate(HOURS):
            tick_vals.append(day_idx * N + h_idx)
            if h_idx % 2 == 0:
                h12    = hour % 12 or 12
                suffix = 'AM' if hour < 12 else 'PM'
                tick_text.append(f'{h12}{suffix}')
            else:
                tick_text.append('')

    fig.update_layout(
        **LAYOUT,
        yaxis_title='Calls per hour',
        xaxis=dict(
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
            range=[-0.5, 5 * N - 0.5],
        ),
    )
    return fig


def plot_14day_daily(fc: pd.DataFrame, hist: pd.DataFrame) -> go.Figure:
    """7 days of actual daily totals + 14 days of predicted daily totals."""
    today    = pd.Timestamp('today').normalize()
    n_actual = 7
    n_pred   = 14

    # Actual daily totals
    act_start  = (today - pd.Timedelta(days=n_actual)).date()
    act_end    = (today - pd.Timedelta(days=1)).date()
    hist_daily = (
        hist[(hist['ds'].dt.date >= act_start) & (hist['ds'].dt.date <= act_end)]
        .groupby(hist['ds'].dt.date)['y'].sum()
        .reset_index()
    )
    hist_daily.columns = ['date', 'calls']
    hist_daily['date'] = pd.to_datetime(hist_daily['date'])

    # Predicted daily totals
    pred_end  = (today + pd.Timedelta(days=n_pred - 1)).date()
    fc_slice  = fc[(fc['ds'].dt.date >= today.date()) & (fc['ds'].dt.date <= pred_end)].copy()
    fc_slice['date'] = pd.to_datetime(fc_slice['ds'].dt.date)
    fc_daily  = fc_slice.groupby('date')[['yhat', 'yhat_lower', 'yhat_upper']].sum().reset_index()

    fig = go.Figure()

    # Confidence band
    if len(fc_daily) > 0:
        fig.add_trace(_band(fc_daily['date'], fc_daily['yhat_lower'], fc_daily['yhat_upper']))

    # Predicted line
    if len(fc_daily) > 0:
        fig.add_trace(go.Scatter(
            x=fc_daily['date'], y=fc_daily['yhat'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color=STEELBLUE, width=2.5),
            marker=dict(size=8, color=STEELBLUE),
            hovertemplate='%{x|%A, %b %d}<br>Predicted: <b>%{y:.0f}</b> calls<extra></extra>',
        ))

    # Actual line
    if len(hist_daily) > 0:
        fig.add_trace(go.Scatter(
            x=hist_daily['date'], y=hist_daily['calls'],
            mode='lines+markers',
            name='Actual',
            line=dict(color=DARK, width=2.5),
            marker=dict(size=8, color=DARK),
            hovertemplate='%{x|%A, %b %d}<br>Actual: <b>%{y:.0f}</b> calls<extra></extra>',
        ))

    # "Today" line
    _today_line(fig, str(today.date()))

    fig.update_layout(
        **LAYOUT,
        yaxis_title='Total calls per day',
        xaxis=dict(
            dtick='D1',
            tickformat='%a<br>%b %d',
            tickfont=dict(size=9),
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


def plot_quarter(fc: pd.DataFrame, hist: pd.DataFrame) -> go.Figure:
    """~5 weeks of actual weekly totals + 13 weeks of predicted weekly totals."""
    today = pd.Timestamp('today').normalize()

    # Actual data — last 35 days, aggregated to weekly totals
    act_start  = today - pd.Timedelta(days=35)
    hist_slice = hist[(hist['ds'] >= act_start) & (hist['ds'] < today)].copy()
    hist_slice['week'] = hist_slice['ds'].dt.to_period('W').dt.start_time
    hist_weekly = hist_slice.groupby('week')['y'].sum().reset_index()
    hist_weekly.columns = ['week', 'calls']

    # Predicted data — next 91 days, aggregated to weekly totals
    pred_end  = today + pd.Timedelta(days=90)
    fc_slice  = fc[(fc['ds'] >= today) & (fc['ds'] <= pred_end)].copy()
    fc_slice['week'] = fc_slice['ds'].dt.to_period('W').dt.start_time
    fc_weekly = fc_slice.groupby('week')[['yhat', 'yhat_lower', 'yhat_upper']].sum().reset_index()

    fig = go.Figure()

    # Confidence band
    if len(fc_weekly) > 0:
        fig.add_trace(_band(fc_weekly['week'], fc_weekly['yhat_lower'], fc_weekly['yhat_upper']))

    # Predicted line
    if len(fc_weekly) > 0:
        fig.add_trace(go.Scatter(
            x=fc_weekly['week'], y=fc_weekly['yhat'],
            mode='lines+markers',
            name='Predicted',
            line=dict(color=STEELBLUE, width=2.5),
            marker=dict(size=8, color=STEELBLUE),
            hovertemplate='Week of %{x|%b %d, %Y}<br>Predicted: <b>%{y:.0f}</b> calls<extra></extra>',
        ))

    # Actual line
    if len(hist_weekly) > 0:
        fig.add_trace(go.Scatter(
            x=hist_weekly['week'], y=hist_weekly['calls'],
            mode='lines+markers',
            name='Actual',
            line=dict(color=DARK, width=2.5),
            marker=dict(size=8, color=DARK),
            hovertemplate='Week of %{x|%b %d, %Y}<br>Actual: <b>%{y:.0f}</b> calls<extra></extra>',
        ))

    # "Today" line
    _today_line(fig, str(today.date()))

    fig.update_layout(
        **LAYOUT,
        yaxis_title='Total calls per week',
        xaxis=dict(
            dtick='M1',
            tickformat='%b %Y',
            tickfont=dict(size=10),
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


# ── App layout ────────────────────────────────────────────────────────────────
st.title('📞 211 Utah — Call Volume Forecast')

with st.spinner('Loading data and training model — takes about 2 minutes on first load…'):
    forecast, hist, record_count, data_through = run_pipeline()

st.caption(
    f"Trained on **{record_count:,}** call records through "
    f"**{data_through.strftime('%B %d, %Y')}**"
)

# Metrics (scoped to 14-day window even though forecast extends 91 days)
today        = pd.Timestamp('today').normalize()
today_fc     = forecast[forecast['ds'].dt.date == today.date()]
fc_14d       = forecast[forecast['ds'].dt.date <= (today + pd.Timedelta(days=13)).date()]
total_14d    = int(fc_14d['yhat'].sum())
fc_14d_daily = fc_14d.copy()
fc_14d_daily['date'] = fc_14d_daily['ds'].dt.date
busiest      = fc_14d_daily.groupby('date')['yhat'].sum().idxmax()

c1, c2, c3 = st.columns(3)
c1.metric("Today's predicted calls",    f"{int(today_fc['yhat'].sum()):,}")
c2.metric("Total calls — next 14 days", f"{total_14d:,}")
c3.metric("Busiest day ahead",          pd.Timestamp(busiest).strftime('%A, %b %d'))

st.divider()

view = st.radio('View', ['3-Day Hourly', '14-Day Daily', 'Quarter Forecast'], horizontal=True)

if view == '3-Day Hourly':
    st.plotly_chart(plot_3day_hourly(forecast, hist), use_container_width=True)
elif view == '14-Day Daily':
    st.plotly_chart(plot_14day_daily(forecast, hist), use_container_width=True)
else:
    st.plotly_chart(plot_quarter(forecast, hist), use_container_width=True)
