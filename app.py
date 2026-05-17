"""
ECOM Trading – Commodity Price Projections Dashboard
Commodities: Coffee (KC=F), Cocoa (CC=F), Cotton (CT=F)
Projection horizon: 3 months | Historical: 5 years
Auto-refresh: every 30 min between 05:00–00:00 MX time
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import time
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# ECOM BRAND PALETTE
# ──────────────────────────────────────────────────────────────────────────────
ECOM_GREEN       = "#3D7A3A"
ECOM_GREEN_LIGHT = "#5BAA56"
ECOM_GREEN_DARK  = "#2A5528"
ECOM_GREY        = "#5A5A5A"
ECOM_GREY_LIGHT  = "#D6D6D6"
ECOM_WHITE       = "#FFFFFF"
ECOM_BLACK       = "#0D0D0D"
ECOM_GOLD        = "#C8A84B"
ECOM_TEAL        = "#2B7A6F"

COMMODITY_COLORS = {
    "Coffee": ECOM_GREEN,
    "Cocoa":  ECOM_TEAL,
    "Cotton": ECOM_GOLD,
}

TICKERS = {
    "Coffee": "KC=F",
    "Cocoa":  "CC=F",
    "Cotton": "CT=F",
}

COMMODITY_UNITS = {
    "Coffee": "cents/lb",
    "Cocoa":  "USD/MT",
    "Cotton": "cents/lb",
}

MACRO_FACTORS = [
    "USD strength (DXY)",
    "El Nino / La Nina weather patterns",
    "Brazil & Vietnam crop reports (Coffee)",
    "Cote d'Ivoire & Ghana output (Cocoa)",
    "US crop reports & USDA WASDE (Cotton)",
    "Global freight & logistics costs",
    "Emerging-market currency volatility",
    "Central bank interest-rate policy (Fed, ECB)",
    "Energy prices impacting fertilizer costs",
    "Geopolitical supply chain disruptions",
]

MX_TZ = pytz.timezone("America/Mexico_City")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECOM | Commodity Projections",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Barlow', sans-serif;
    background-color: #F4F5F0;
    color: {ECOM_GREY};
  }}

  section[data-testid="stSidebar"] {{
    background: linear-gradient(175deg, {ECOM_GREEN_DARK} 0%, {ECOM_GREEN} 60%, {ECOM_TEAL} 100%);
    color: white;
  }}
  section[data-testid="stSidebar"] * {{
    color: white !important;
  }}
  section[data-testid="stSidebar"] .stButton > button {{
    background: white;
    color: {ECOM_GREEN_DARK} !important;
    font-weight: 700;
    border: none;
    border-radius: 4px;
    width: 100%;
    padding: 0.6rem;
    font-size: 0.95rem;
    transition: all 0.2s;
  }}
  section[data-testid="stSidebar"] .stButton > button:hover {{
    background: {ECOM_GOLD};
    color: white !important;
  }}

  .ecom-header {{
    background: linear-gradient(90deg, {ECOM_GREEN_DARK} 0%, {ECOM_GREEN} 70%, {ECOM_TEAL} 100%);
    padding: 1rem 2rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
  }}
  .ecom-header h1 {{
    font-family: 'Barlow Condensed', sans-serif;
    color: white;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .ecom-header span {{
    font-size: 0.8rem;
    color: rgba(255,255,255,0.75);
    font-weight: 300;
  }}

  .kpi-card {{
    background: white;
    border-left: 4px solid {ECOM_GREEN};
    border-radius: 6px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .kpi-label {{ font-size: 0.72rem; color: {ECOM_GREY}; text-transform: uppercase; letter-spacing: 0.06em; }}
  .kpi-value {{ font-size: 1.6rem; font-weight: 700; color: {ECOM_GREEN_DARK}; }}
  .kpi-delta {{ font-size: 0.82rem; margin-top: 0.15rem; }}
  .delta-up   {{ color: {ECOM_GREEN}; }}
  .delta-down {{ color: #C0392B; }}

  .result-box {{
    background: {ECOM_BLACK};
    color: {ECOM_WHITE};
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-top: 0.8rem;
    border: 1px solid {ECOM_GREEN_DARK};
  }}
  .result-box h3 {{
    color: {ECOM_GREEN_LIGHT};
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }}

  .section-title {{
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: {ECOM_GREEN_DARK};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 2px solid {ECOM_GREEN};
    padding-bottom: 0.3rem;
    margin-bottom: 0.8rem;
  }}

  .stTabs [data-baseweb="tab-list"] {{
    background: white;
    border-radius: 6px;
    padding: 4px;
    gap: 4px;
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 4px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
  }}
  .stTabs [aria-selected="true"] {{
    background: {ECOM_GREEN} !important;
    color: white !important;
  }}

  footer {{ visibility: hidden; }}
  #MainMenu {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────

def get_mx_time():
    return datetime.now(MX_TZ)


def is_refresh_window():
    return get_mx_time().hour >= 5


def _squeeze(obj):
    """Ensure a DataFrame column comes back as a plain 1-D Series."""
    if isinstance(obj, pd.DataFrame):
        obj = obj.squeeze(axis=1)
    return obj


def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance >=0.2 returns MultiIndex columns like ('Close','KC=F').
    Flatten to single-level ('Close').
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    # Drop duplicate column names that can appear after flattening
    df = df.loc[:, ~df.columns.duplicated()]
    return df


@st.cache_data(ttl=1800)
def fetch_data(ticker: str, period_years: int = 5) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=period_years * 365)
    try:
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        raw = _flatten_df(raw)

        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
        df = raw[needed].copy()
        df.index = pd.to_datetime(df.index)

        # Remove timezone info to avoid mixing issues
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.dropna(subset=["Close"], inplace=True)
        df.sort_index(inplace=True)
        return df

    except Exception as exc:
        st.warning(f"Could not fetch {ticker}: {exc}")
        return pd.DataFrame()


def get_close(df: pd.DataFrame) -> pd.Series:
    """Return Close as a guaranteed 1-D Series."""
    return _squeeze(df["Close"]).astype(float)


def compute_projections(df: pd.DataFrame, months: int = 3) -> dict:
    if df.empty or len(df) < 60:
        return {}

    close  = get_close(df)
    series = close.resample("W").last().dropna().astype(float)
    n_weeks = months * 4

    last_val = float(series.iloc[-1])

    # Holt-Winters
    try:
        hw_fit = ExponentialSmoothing(
            series, trend="add", seasonal="add", seasonal_periods=52
        ).fit(optimized=True, use_brute=False)
        hw_fc = hw_fit.forecast(n_weeks).values.astype(float)
    except Exception:
        hw_fc = np.full(n_weeks, last_val)

    # ARIMA(2,1,2)
    try:
        arima_fc = ARIMA(series, order=(2, 1, 2)).fit().forecast(n_weeks).values.astype(float)
    except Exception:
        arima_fc = np.full(n_weeks, last_val)

    # Polynomial Trend (degree 3)
    try:
        x      = np.arange(len(series)).reshape(-1, 1)
        pf     = PolynomialFeatures(degree=3)
        reg    = LinearRegression().fit(pf.fit_transform(x), series.values)
        x_fut  = np.arange(len(series), len(series) + n_weeks).reshape(-1, 1)
        poly_fc = reg.predict(pf.transform(x_fut)).astype(float)
    except Exception:
        poly_fc = np.full(n_weeks, last_val)

    ensemble    = hw_fc * 0.45 + arima_fc * 0.35 + poly_fc * 0.20
    rolling_std = float(series.rolling(52).std().dropna().iloc[-1])
    ci_upper    = ensemble + 1.96 * rolling_std
    ci_lower    = ensemble - 1.96 * rolling_std

    fut_dates  = pd.date_range(
        series.index[-1] + pd.Timedelta(weeks=1), periods=n_weeks, freq="W"
    )
    proj_price = float(ensemble[-1])

    return {
        "historical": series,
        "dates":      fut_dates,
        "hw":         hw_fc,
        "arima":      arima_fc,
        "poly":       poly_fc,
        "ensemble":   ensemble,
        "ci_upper":   ci_upper,
        "ci_lower":   ci_lower,
        "last_price": last_val,
        "proj_price": proj_price,
        "pct_change": (proj_price - last_val) / last_val * 100,
    }


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


def format_delta(pct: float) -> str:
    arrow = "&#9650;" if pct >= 0 else "&#9660;"
    cls   = "delta-up" if pct >= 0 else "delta-down"
    return f'<span class="{cls}">{arrow} {abs(pct):.2f}%</span>'


def base_layout(title="", height=420) -> dict:
    return dict(
        title=title,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="#F8F9F5",
        font=dict(family="Barlow, sans-serif", color=ECOM_GREY, size=12),
        title_font=dict(family="Barlow Condensed, sans-serif", size=16, color=ECOM_GREEN_DARK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=30, t=50, b=40),
        xaxis=dict(gridcolor=ECOM_GREY_LIGHT, showgrid=True),
        yaxis=dict(gridcolor=ECOM_GREY_LIGHT, showgrid=True),
    )


# ──────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────────────────────────────────────

def chart_price_forecast(df, proj, commodity, unit):
    hist  = proj["historical"]
    dates = proj["dates"]
    col   = COMMODITY_COLORS[commodity]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values,
        name="Historical (weekly)",
        line=dict(color=col, width=1.8),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_to_rgb(col)},0.08)",
        hovertemplate="%{x|%b %d %Y}  %{y:.2f} " + unit + "<extra></extra>",
    ))
    # CI ribbon
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(proj["ci_upper"]) + list(proj["ci_lower"][::-1]),
        fill="toself",
        fillcolor=f"rgba({_hex_to_rgb(col)},0.15)",
        line=dict(width=0),
        name="95% CI",
        hoverinfo="skip",
    ))
    for key, mc, dash, label in [
        ("hw",    ECOM_GREEN_LIGHT, "dot",     "Holt-Winters"),
        ("arima", ECOM_TEAL,        "dash",    "ARIMA(2,1,2)"),
        ("poly",  ECOM_GOLD,        "dashdot", "Poly Trend"),
    ]:
        fig.add_trace(go.Scatter(
            x=dates, y=proj[key],
            name=label,
            line=dict(color=mc, width=1.4, dash=dash),
            hovertemplate="%{x|%b %d %Y}  %{y:.2f} " + unit + "<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=dates, y=proj["ensemble"],
        name="Ensemble",
        line=dict(color=ECOM_GREEN_DARK, width=2.8),
        hovertemplate="%{x|%b %d %Y}  %{y:.2f} " + unit + "<extra></extra>",
    ))
    # Vertical "Today" line
    today_str = str(hist.index[-1].date())
    fig.add_vline(x=today_str, line_dash="dash", line_color=ECOM_GREY, opacity=0.5)
    fig.add_annotation(
        x=today_str, y=float(hist.values.max()),
        text="Today", showarrow=False,
        font=dict(size=10, color=ECOM_GREY), xshift=12,
    )
    fig.update_layout(**base_layout(
        f"{commodity} – {len(df)//252}-Year History + Projection  ({unit})", 470
    ))
    return fig


def chart_candlestick(df, commodity, unit):
    # FIX: pandas 2.x removed DataFrame.last() – use boolean index instead
    cutoff = df.index.max() - pd.Timedelta(days=180)
    recent = df[df.index >= cutoff].copy()

    fig = go.Figure(go.Candlestick(
        x=recent.index,
        open=_squeeze(recent["Open"]).astype(float),
        high=_squeeze(recent["High"]).astype(float),
        low=_squeeze(recent["Low"]).astype(float),
        close=_squeeze(recent["Close"]).astype(float),
        increasing_line_color=ECOM_GREEN,
        decreasing_line_color="#C0392B",
        name=commodity,
    ))
    fig.update_layout(**base_layout(f"{commodity} – Last 6 Months  ({unit})", 420))
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def chart_monthly_returns(df, commodity):
    close   = get_close(df)
    monthly = close.resample("ME").last().pct_change().dropna() * 100
    colors  = [ECOM_GREEN if v >= 0 else "#C0392B" for v in monthly.values]
    fig = go.Figure(go.Bar(
        x=monthly.index, y=monthly.values,
        marker_color=colors,
        hovertemplate="%{x|%b %Y}  %{y:.2f}%<extra></extra>",
        name="Monthly Return",
    ))
    fig.update_layout(**base_layout(f"{commodity} – Monthly Returns (%)", 330))
    return fig


def chart_vol_heatmap(datasets: dict):
    vol_data = {}
    for name, df in datasets.items():
        if df.empty:
            continue
        close = get_close(df)
        vol_data[name] = close.pct_change().rolling(30).std() * np.sqrt(252) * 100

    if not vol_data:
        return go.Figure()

    vol_df = pd.DataFrame(vol_data).dropna().resample("ME").mean()
    fig = go.Figure(go.Heatmap(
        x=vol_df.index,
        y=list(vol_df.columns),
        z=vol_df.T.values,
        colorscale=[[0.0, "#EEF5ED"], [0.5, ECOM_GREEN], [1.0, ECOM_GREEN_DARK]],
        hovertemplate="%{x|%b %Y}  %{y}: %{z:.1f}%<extra></extra>",
        colorbar=dict(title="Ann. Vol %"),
    ))
    fig.update_layout(**base_layout("Annualised Volatility Heatmap (30-day rolling)", 320))
    return fig


def chart_corr_heatmap(datasets: dict):
    ret_data = {}
    for name, df in datasets.items():
        if df.empty:
            continue
        close = get_close(df)
        ret_data[name] = close.resample("W").last().pct_change().dropna()

    if len(ret_data) < 2:
        return go.Figure()

    corr   = pd.DataFrame(ret_data).dropna().corr()
    labels = list(corr.columns)
    fig = go.Figure(go.Heatmap(
        x=labels, y=labels, z=corr.values,
        colorscale=[[0, "#C0392B"], [0.5, ECOM_GREY_LIGHT], [1, ECOM_GREEN]],
        zmin=-1, zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        hovertemplate="%{x} vs %{y}  r=%{z:.2f}<extra></extra>",
        colorbar=dict(title="Pearson r"),
    ))
    fig.update_layout(**base_layout("Weekly Return Correlation Matrix", 340))
    return fig


def chart_seasonality(df, commodity, unit):
    close = get_close(df)
    avg   = close.groupby(close.index.month).mean()
    avg   = avg.reindex(range(1, 13), fill_value=0)
    mnths = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mnths, y=avg.values,
        marker_color=ECOM_GREEN,
        name="Avg Price",
        hovertemplate="%{x}: %{y:.2f} " + unit + "<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=mnths, y=avg.values,
        mode="lines+markers",
        line=dict(color=ECOM_GOLD, width=2),
        marker=dict(size=7),
        name="Trend",
    ))
    fig.update_layout(**base_layout(f"{commodity} – Avg Price by Month (Seasonality)", 320))
    return fig


def chart_portfolio_donut(proj_data: dict):
    labels = [c for c in proj_data if proj_data[c]]
    values = [proj_data[c]["proj_price"] for c in labels]
    colors = [COMMODITY_COLORS[c] for c in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.52,
        marker=dict(colors=colors),
        hovertemplate="%{label}  Proj: %{value:.2f}  %{percent}<extra></extra>",
    ))
    fig.update_layout(**base_layout("Relative Projected Price Weight", 340))
    return fig


def chart_current_vs_proj(proj_data: dict):
    comms = [c for c in proj_data if proj_data[c]]
    fig   = go.Figure()
    fig.add_trace(go.Bar(
        name="Current Price", x=comms,
        y=[proj_data[c]["last_price"] for c in comms],
        marker_color=ECOM_GREY_LIGHT,
        hovertemplate="%{x}  Current: %{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="3M Projected", x=comms,
        y=[proj_data[c]["proj_price"] for c in comms],
        marker_color=ECOM_GREEN,
        hovertemplate="%{x}  Projected: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(**base_layout("Current vs. 3-Month Projected Prices", 340), barmode="group")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────────────────────
for key, val in [("data", {}), ("proj", {}), ("last_update", None), ("calculated", False)]:
    if key not in st.session_state:
        st.session_state[key] = val

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align:center;padding:0.8rem 0 1.2rem;">
            <div style="font-family:'Barlow Condensed',sans-serif;font-size:1.7rem;
                        font-weight:700;letter-spacing:0.08em;color:white;">&#127807; ECOM</div>
            <div style="font-size:0.72rem;color:rgba(255,255,255,0.7);letter-spacing:0.1em;">
                COMMODITY PROJECTIONS
            </div>
            <div style="font-size:0.65rem;color:rgba(255,255,255,0.5);margin-top:4px;">
                Americas Region &middot; Dallas &middot; Mexico Entities
            </div>
        </div>
        <hr style="border-color:rgba(255,255,255,0.2);margin-bottom:1rem;">
    """, unsafe_allow_html=True)

    st.markdown("**COMMODITY SELECTION**")
    selected_commodities = st.multiselect(
        "Select Commodities",
        options=list(TICKERS.keys()),
        default=list(TICKERS.keys()),
    )

    st.markdown("**PROJECTION SETTINGS**")
    proj_months = st.slider("Projection Horizon (months)", 1, 6, 3)
    hist_years  = st.slider("Historical Data (years)", 1, 5, 5)

    st.markdown("**MODEL WEIGHTS**")
    w_hw    = st.slider("Holt-Winters",  0.0, 1.0, 0.45, 0.05)
    w_arima = st.slider("ARIMA",         0.0, 1.0, 0.35, 0.05)
    w_poly  = round(max(0.0, 1.0 - w_hw - w_arima), 2)
    st.caption(f"Polynomial Trend weight auto: **{w_poly}**")

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("&#9654;  RUN CALCULATION", use_container_width=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    now_mx = get_mx_time()
    st.markdown(f"""
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.65);line-height:1.8;">
            MX Time: {now_mx.strftime('%H:%M')} CST<br>
            Auto-refresh: {'Active' if is_refresh_window() else 'Off (05:00-00:00 only)'}<br>
            Cache TTL: 30 min
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.last_update:
        st.caption(f"Last run: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M')}")

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
    <div class="ecom-header">
        <div>
            <h1>Commodity Price Projections</h1>
            <span>Coffee &middot; Cocoa &middot; Cotton &nbsp;|&nbsp;
                  {hist_years}-Year History &nbsp;|&nbsp; {proj_months}-Month Outlook</span>
        </div>
        <div style="text-align:right;">
            <span style="color:rgba(255,255,255,0.9);font-size:0.9rem;font-weight:600;">
                Grupo ECOM Trading</span><br>
            <span>Americas Region &ndash; Dallas</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# CALCULATION
# ──────────────────────────────────────────────────────────────────────────────
if run_btn or (is_refresh_window() and not st.session_state.calculated):
    with st.spinner("Fetching market data and computing projections..."):
        data_store = {}
        proj_store = {}
        for commodity in selected_commodities:
            df = fetch_data(TICKERS[commodity], hist_years)
            data_store[commodity] = df
            if not df.empty:
                proj = compute_projections(df, proj_months)
                if proj:
                    total = w_hw + w_arima + w_poly or 1.0
                    proj["ensemble"] = (
                        proj["hw"]    * (w_hw    / total) +
                        proj["arima"] * (w_arima / total) +
                        proj["poly"]  * (w_poly  / total)
                    )
                    proj["proj_price"] = float(proj["ensemble"][-1])
                    proj["pct_change"] = (
                        (proj["proj_price"] - proj["last_price"]) / proj["last_price"] * 100
                    )
                proj_store[commodity] = proj
            else:
                proj_store[commodity] = {}

        st.session_state.data        = data_store
        st.session_state.proj        = proj_store
        st.session_state.last_update = datetime.now(MX_TZ)
        st.session_state.calculated  = True

if not st.session_state.calculated:
    st.info("Select commodities and click  RUN CALCULATION  to generate projections.")
    st.stop()

data_store = st.session_state.data
proj_store = st.session_state.proj

# ──────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Market Snapshot</div>', unsafe_allow_html=True)
kpi_cols = st.columns(max(len(selected_commodities), 1))
for i, commodity in enumerate(selected_commodities):
    proj = proj_store.get(commodity, {})
    unit = COMMODITY_UNITS[commodity]
    if not proj:
        kpi_cols[i].warning(f"{commodity}: data unavailable")
        continue
    kpi_cols[i].markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{commodity} &nbsp;({TICKERS[commodity]}) &nbsp;{unit}</div>
            <div class="kpi-value">{proj['last_price']:.2f}</div>
            <div class="kpi-delta">
                {proj_months}M &rarr; <b>{proj['proj_price']:.2f}</b>
                &nbsp; {format_delta(proj['pct_change'])}
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# BLACK RESULT BOX
# ──────────────────────────────────────────────────────────────────────────────
lines = []
for c in selected_commodities:
    p = proj_store.get(c, {})
    if not p:
        continue
    unit  = COMMODITY_UNITS[c]
    arrow = "&#9650;" if p["pct_change"] >= 0 else "&#9660;"
    lines.append(
        f"<b>{c}</b> ({TICKERS[c]}): "
        f"{p['last_price']:.2f} {unit} &rarr; "
        f"<b>{p['proj_price']:.2f} {unit}</b> "
        f"({arrow} {abs(p['pct_change']):.2f}%)"
    )

st.markdown(f"""
    <div class="result-box">
        <h3>Projection Summary &ndash; {proj_months}-Month Outlook</h3>
        {"<br>".join(lines)}
        <br><br>
        <span style="color:{ECOM_GREY_LIGHT};font-size:0.82rem;">
        Ensemble: Holt-Winters ({w_hw:.0%}) &middot; ARIMA(2,1,2) ({w_arima:.0%})
        &middot; Poly Trend ({w_poly:.0%})<br>
        Key macro: {' &middot; '.join(MACRO_FACTORS[:3])} ...<br>
        Legal entities: Mexico &nbsp;|&nbsp; Reporting: Americas – Dallas, Grupo ECOM Trading
        </span>
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📈 Price & Forecast",
    "🕯 Candlestick",
    "📊 Returns & Volatility",
    "🌡 Heatmaps",
    "🧩 Portfolio View",
    "🌿 Macro Factors",
])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with t1:
    for commodity in selected_commodities:
        df   = data_store.get(commodity, pd.DataFrame())
        proj = proj_store.get(commodity, {})
        unit = COMMODITY_UNITS[commodity]
        if df.empty or not proj:
            st.warning(f"No data for {commodity}.")
            continue
        st.plotly_chart(chart_price_forecast(df, proj, commodity, unit),
                        use_container_width=True, key=f"pf_{commodity}")
        st.plotly_chart(chart_seasonality(df, commodity, unit),
                        use_container_width=True, key=f"sea_{commodity}")
        st.divider()

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with t2:
    for commodity in selected_commodities:
        df = data_store.get(commodity, pd.DataFrame())
        if df.empty:
            st.warning(f"No data for {commodity}.")
            continue
        st.plotly_chart(chart_candlestick(df, commodity, COMMODITY_UNITS[commodity]),
                        use_container_width=True, key=f"cs_{commodity}")
        st.divider()

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with t3:
    for commodity in selected_commodities:
        df = data_store.get(commodity, pd.DataFrame())
        if df.empty:
            continue
        st.plotly_chart(chart_monthly_returns(df, commodity),
                        use_container_width=True, key=f"mr_{commodity}")

    has_data = any(not data_store.get(c, pd.DataFrame()).empty for c in selected_commodities)
    if has_data:
        st.markdown('<div class="section-title">Rolling 30-Day Annualised Volatility</div>',
                    unsafe_allow_html=True)
        vfig = go.Figure()
        for commodity in selected_commodities:
            df = data_store.get(commodity, pd.DataFrame())
            if df.empty:
                continue
            close = get_close(df)
            vol   = close.pct_change().rolling(30).std() * np.sqrt(252) * 100
            vfig.add_trace(go.Scatter(
                x=vol.index, y=vol.values, name=commodity,
                line=dict(color=COMMODITY_COLORS[commodity], width=1.8),
                hovertemplate="%{x|%b %d %Y}  Vol: %{y:.1f}%<extra></extra>",
            ))
        vfig.update_layout(**base_layout("Rolling 30-Day Annualised Volatility (%)", 380))
        st.plotly_chart(vfig, use_container_width=True, key="vol_line")

# ── Tab 4 ─────────────────────────────────────────────────────────────────────
with t4:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_vol_heatmap(data_store),
                        use_container_width=True, key="vh")
    with c2:
        st.plotly_chart(chart_corr_heatmap(data_store),
                        use_container_width=True, key="ch")

    st.markdown('<div class="section-title">Normalised Price Index (Base = 100)</div>',
                unsafe_allow_html=True)
    nfig = go.Figure()
    for commodity in selected_commodities:
        df = data_store.get(commodity, pd.DataFrame())
        if df.empty:
            continue
        close  = get_close(df)
        weekly = close.resample("W").last().dropna()
        norm   = (weekly / float(weekly.iloc[0])) * 100
        nfig.add_trace(go.Scatter(
            x=norm.index, y=norm.values, name=commodity,
            line=dict(color=COMMODITY_COLORS[commodity], width=2),
            hovertemplate="%{x|%b %Y}  Index: %{y:.1f}<extra></extra>",
        ))
    nfig.update_layout(**base_layout("5-Year Normalised Price Index", 380))
    st.plotly_chart(nfig, use_container_width=True, key="norm")

# ── Tab 5 ─────────────────────────────────────────────────────────────────────
with t5:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(chart_portfolio_donut(proj_store),
                        use_container_width=True, key="donut")
    with c2:
        st.plotly_chart(chart_current_vs_proj(proj_store),
                        use_container_width=True, key="cvp")

    st.markdown('<div class="section-title">Model Comparison Table</div>', unsafe_allow_html=True)
    rows = []
    for commodity in selected_commodities:
        proj = proj_store.get(commodity, {})
        if not proj:
            continue
        rows.append({
            "Commodity":    commodity,
            "Ticker":       TICKERS[commodity],
            "Unit":         COMMODITY_UNITS[commodity],
            "Current":      round(proj["last_price"], 2),
            "Holt-Winters": round(float(proj["hw"][-1]), 2),
            "ARIMA(2,1,2)": round(float(proj["arima"][-1]), 2),
            "Poly Trend":   round(float(proj["poly"][-1]), 2),
            "Ensemble":     round(proj["proj_price"], 2),
            "Delta %":      f"{proj['pct_change']:+.2f}%",
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows).set_index("Commodity"),
            use_container_width=True, height=180,
        )

# ── Tab 6 ─────────────────────────────────────────────────────────────────────
with t6:
    st.markdown('<div class="section-title">Macroeconomic Factors Considered</div>',
                unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    for idx, factor in enumerate(MACRO_FACTORS):
        icon = ("Coffee" in factor and "&#9749;") or \
               ("Cocoa"  in factor and "&#127849;") or \
               ("Cotton" in factor and "&#127807;") or "&#127758;"
        (fc1 if idx % 2 == 0 else fc2).markdown(f"""
            <div style="background:white;border-left:4px solid {ECOM_GREEN};
                        border-radius:4px;padding:0.7rem 1rem;margin-bottom:0.6rem;
                        box-shadow:0 1px 4px rgba(0,0,0,0.06);">
                <span style="font-weight:600;color:{ECOM_GREEN_DARK};">
                    {icon}&nbsp;&nbsp;{factor}
                </span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="result-box" style="margin-top:1.2rem;">
            <h3>Model Disclaimer</h3>
            Projections use Holt-Winters ESM, ARIMA(2,1,2), and Polynomial Regression on
            publicly available futures prices from <b>yfinance</b>. These are
            <b>indicative forecasts only</b> and do not constitute financial advice.
            Prices in the contract's native currency (USD).<br><br>
            <span style="color:{ECOM_GREY_LIGHT};">
            Legal entities: Mexico &nbsp;|&nbsp; Reporting: Americas (Dallas)
            &nbsp;|&nbsp; Grupo ECOM Trading &copy; {datetime.now().year}
            </span>
        </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# AUTO-REFRESH (30 min, only during active window)
# ──────────────────────────────────────────────────────────────────────────────
if is_refresh_window():
    time.sleep(1800)
    st.rerun()
