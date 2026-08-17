import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Blue-Chip Value Scanner",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Curated large, established U.S.-listed companies.
# This is intentionally fixed so that the model is transparent and reproducible.
UNIVERSE = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "META": "Meta Platforms",
    "BRK-B": "Berkshire Hathaway",
    "JPM": "JPMorgan Chase",
    "V": "Visa",
    "MA": "Mastercard",
    "LLY": "Eli Lilly",
    "AVGO": "Broadcom",
    "WMT": "Walmart",
    "COST": "Costco",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble",
    "HD": "Home Depot",
    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "CVX": "Chevron",
    "BAC": "Bank of America",
    "ORCL": "Oracle",
    "CRM": "Salesforce",
    "CSCO": "Cisco",
    "IBM": "IBM",
    "ACN": "Accenture",
    "MCD": "McDonald's",
    "DIS": "Walt Disney",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "QCOM": "Qualcomm",
    "TXN": "Texas Instruments",
    "INTU": "Intuit",
    "AMGN": "Amgen",
    "CAT": "Caterpillar",
    "GE": "GE Aerospace",
    "UNH": "UnitedHealth",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "AXP": "American Express",
    "SPGI": "S&P Global",
    "BLK": "BlackRock",
    "LOW": "Lowe's",
    "NEE": "NextEra Energy",
    "RTX": "RTX",
    "HON": "Honeywell",
    "UPS": "UPS",
}

FINANCIAL_SECTORS = {"Financial Services", "Financials"}

def finite(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else np.nan
    except Exception:
        return np.nan

def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def linear_score(x, bad, good):
    """0 at bad, 100 at good; works in either direction."""
    x = finite(x)
    if pd.isna(x):
        return 50.0
    if good == bad:
        return 50.0
    return clamp((x - bad) / (good - bad) * 100.0)

def safe_get(info, *keys):
    for key in keys:
        if key in info and info[key] is not None:
            return info[key]
    return np.nan

@st.cache_data(ttl=3600, show_spinner=False)
def download_prices(tickers):
    raw = yf.download(
        tickers=list(tickers),
        period="3y",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw is None or raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            return raw["Close"].copy()
        if "Close" in raw.columns.get_level_values(1):
            return raw.xs("Close", axis=1, level=1).copy()
    # Single-ticker fallback
    if "Close" in raw.columns:
        out = raw[["Close"]].copy()
        out.columns = [list(tickers)[0]]
        return out
    return pd.DataFrame()

@st.cache_data(ttl=21600, show_spinner=False)
def get_info(symbol):
    try:
        return yf.Ticker(symbol).get_info() or {}
    except Exception:
        try:
            return yf.Ticker(symbol).info or {}
        except Exception:
            return {}

def price_features(series):
    s = series.dropna()
    if s.empty:
        return {}
    current = float(s.iloc[-1])
    one_year = s.iloc[-252:] if len(s) >= 252 else s
    high_52 = float(one_year.max())
    low_52 = float(one_year.min())
    ma50 = float(s.iloc[-50:].mean()) if len(s) >= 20 else np.nan
    ma200 = float(s.iloc[-200:].mean()) if len(s) >= 60 else np.nan

    ret_1y = np.nan
    if len(s) > 252 and s.iloc[-253] > 0:
        ret_1y = current / float(s.iloc[-253]) - 1
    elif len(s) > 30 and s.iloc[0] > 0:
        ret_1y = current / float(s.iloc[0]) - 1

    drawdown = current / high_52 - 1 if high_52 else np.nan
    return {
        "price": current,
        "high_52": high_52,
        "low_52": low_52,
        "ma50": ma50,
        "ma200": ma200,
        "return_1y": ret_1y,
        "drawdown_52": drawdown,
    }

def estimate_fair_value(price, info):
    """
    Transparent heuristic fair-value range.
    Uses:
      1) normalized forward-P/E method,
      2) free-cash-flow yield method,
      3) analyst mean target as a lower-weight reference when available.
    It is deliberately capped to reduce extreme estimates.
    """
    forward_pe = finite(safe_get(info, "forwardPE"))
    trailing_pe = finite(safe_get(info, "trailingPE"))
    earnings_growth = finite(safe_get(info, "earningsGrowth"))
    revenue_growth = finite(safe_get(info, "revenueGrowth"))
    roe = finite(safe_get(info, "returnOnEquity"))
    margin = finite(safe_get(info, "profitMargins"))
    market_cap = finite(safe_get(info, "marketCap"))
    fcf = finite(safe_get(info, "freeCashflow"))
    target = finite(safe_get(info, "targetMeanPrice"))
    sector = str(safe_get(info, "sector"))

    # Reasonable earnings multiple based on growth + quality, not on the current multiple.
    growth = np.nanmean([v for v in [earnings_growth, revenue_growth] if not pd.isna(v)])
    if pd.isna(growth):
        growth = 0.05
    quality_bonus = 0.0
    if not pd.isna(roe):
        quality_bonus += clamp((roe - 0.10) * 20, -2, 4)
    if not pd.isna(margin):
        quality_bonus += clamp((margin - 0.10) * 10, -1, 3)

    if sector in FINANCIAL_SECTORS:
        base_pe = 13.0
        fair_pe = clamp(base_pe + growth * 18 + quality_bonus * 0.6, 9, 20)
    else:
        base_pe = 17.0
        fair_pe = clamp(base_pe + growth * 22 + quality_bonus, 11, 32)

    estimates = []

    pe = forward_pe if not pd.isna(forward_pe) and forward_pe > 0 else trailing_pe
    if not pd.isna(pe) and pe > 0:
        pe_est = price * fair_pe / pe
        if 0.45 * price <= pe_est <= 2.2 * price:
            estimates.append(("Normalized P/E", pe_est, 0.50))

    if (
        sector not in FINANCIAL_SECTORS
        and not pd.isna(market_cap) and market_cap > 0
        and not pd.isna(fcf) and fcf > 0
    ):
        current_fcf_yield = fcf / market_cap
        # Quality companies often trade around a 4%-6% FCF yield.
        target_yield = 0.05
        if growth > 0.12:
            target_yield = 0.045
        elif growth < 0.02:
            target_yield = 0.06
        fcf_est = price * current_fcf_yield / target_yield
        if 0.45 * price <= fcf_est <= 2.2 * price:
            estimates.append(("FCF yield", fcf_est, 0.30))

    if not pd.isna(target) and target > 0 and 0.45 * price <= target <= 2.2 * price:
        estimates.append(("Analyst target", target, 0.20))

    if not estimates:
        return np.nan, np.nan, np.nan, "Insufficient valuation data"

    # Renormalize available weights.
    total_w = sum(w for _, _, w in estimates)
    fair = sum(v * w for _, v, w in estimates) / total_w
    fair = clamp(fair, 0.55 * price, 1.8 * price)
    low = fair * 0.90
    high = fair * 1.10
    methods = ", ".join(name for name, _, _ in estimates)
    return fair, low, high, methods

def score_stock(symbol, company, p, info):
    price = finite(p.get("price"))
    if pd.isna(price) or price <= 0:
        return None

    sector = str(safe_get(info, "sector"))
    forward_pe = finite(safe_get(info, "forwardPE"))
    trailing_pe = finite(safe_get(info, "trailingPE"))
    peg = finite(safe_get(info, "pegRatio", "trailingPegRatio"))
    profit_margin = finite(safe_get(info, "profitMargins"))
    roe = finite(safe_get(info, "returnOnEquity"))
    debt_to_equity = finite(safe_get(info, "debtToEquity"))
    revenue_growth = finite(safe_get(info, "revenueGrowth"))
    earnings_growth = finite(safe_get(info, "earningsGrowth"))
    fcf = finite(safe_get(info, "freeCashflow"))
    market_cap = finite(safe_get(info, "marketCap"))

    fair, fair_low, fair_high, methods = estimate_fair_value(price, info)
    discount = price / fair - 1 if not pd.isna(fair) and fair > 0 else np.nan

    # 1) Valuation: estimated discount + basic multiples/yield sanity checks.
    discount_score = linear_score(discount, 0.10, -0.30)
    pe = forward_pe if not pd.isna(forward_pe) else trailing_pe
    pe_score = linear_score(pe, 35, 12) if not pd.isna(pe) and pe > 0 else 50
    peg_score = linear_score(peg, 3.0, 1.0) if not pd.isna(peg) and peg > 0 else 50

    fcf_yield = np.nan
    if not pd.isna(fcf) and not pd.isna(market_cap) and market_cap > 0:
        fcf_yield = fcf / market_cap
    fcf_yield_score = linear_score(fcf_yield, 0.02, 0.07) if sector not in FINANCIAL_SECTORS else 50
    valuation_score = np.mean([discount_score, pe_score, peg_score, fcf_yield_score])

    # 2) Price dislocation: reward meaningful drawdowns, but not endlessly.
    drawdown = finite(p.get("drawdown_52"))
    drawdown_score = linear_score(drawdown, -0.05, -0.30)
    # Very severe drawdowns can indicate a broken thesis; cap contribution.
    if not pd.isna(drawdown) and drawdown < -0.45:
        drawdown_score = min(drawdown_score, 80)
    below_ma200 = price / p["ma200"] - 1 if not pd.isna(p.get("ma200", np.nan)) and p["ma200"] else np.nan
    ma_dislocation = linear_score(below_ma200, 0.10, -0.20)
    dislocation_score = np.mean([drawdown_score, ma_dislocation])

    # 3) Business quality.
    margin_score = linear_score(profit_margin, 0.05, 0.25)
    roe_score = linear_score(roe, 0.08, 0.25)
    if sector in FINANCIAL_SECTORS:
        debt_score = 60  # D/E is structurally different for financial institutions.
    else:
        debt_score = linear_score(debt_to_equity, 200, 30)
    positive_fcf_score = 90 if not pd.isna(fcf) and fcf > 0 else (20 if not pd.isna(fcf) else 50)
    quality_score = np.mean([margin_score, roe_score, debt_score, positive_fcf_score])

    # 4) Growth.
    rev_score = linear_score(revenue_growth, -0.05, 0.15)
    earn_score = linear_score(earnings_growth, -0.10, 0.20)
    growth_score = np.mean([rev_score, earn_score])

    # 5) Momentum: smaller weight; avoids catching every falling knife.
    above_50 = price / p["ma50"] - 1 if not pd.isna(p.get("ma50", np.nan)) and p["ma50"] else np.nan
    above_200 = price / p["ma200"] - 1 if not pd.isna(p.get("ma200", np.nan)) and p["ma200"] else np.nan
    momentum_score = np.mean([
        linear_score(above_50, -0.15, 0.08),
        linear_score(above_200, -0.25, 0.10),
    ])

    base = (
        valuation_score * 0.30
        + dislocation_score * 0.25
        + quality_score * 0.25
        + growth_score * 0.15
        + momentum_score * 0.05
    )

    # Value-trap penalties.
    penalty = 0
    flags = []
    if not pd.isna(revenue_growth) and revenue_growth < -0.08:
        penalty += 6
        flags.append("revenue shrinking")
    if not pd.isna(earnings_growth) and earnings_growth < -0.20:
        penalty += 7
        flags.append("earnings shrinking")
    if sector not in FINANCIAL_SECTORS and not pd.isna(debt_to_equity) and debt_to_equity > 250:
        penalty += 5
        flags.append("high debt")
    if not pd.isna(fcf) and fcf <= 0:
        penalty += 7
        flags.append("negative FCF")
    if not pd.isna(profit_margin) and profit_margin < 0:
        penalty += 7
        flags.append("negative margin")

    opportunity = clamp(base - penalty)

    if opportunity >= 80:
        signal = "🟢 Strong candidate"
    elif opportunity >= 70:
        signal = "🟢 Attractive"
    elif opportunity >= 60:
        signal = "🟡 Watch"
    else:
        signal = "⚪ Not compelling"

    return {
        "Ticker": symbol,
        "Company": company,
        "Sector": sector if sector != "nan" else "",
        "Price": price,
        "Fair Value": fair,
        "Fair Low": fair_low,
        "Fair High": fair_high,
        "Discount": discount,
        "52W Drawdown": drawdown,
        "Forward P/E": forward_pe,
        "FCF Yield": fcf_yield,
        "Revenue Growth": revenue_growth,
        "Earnings Growth": earnings_growth,
        "ROE": roe,
        "Profit Margin": profit_margin,
        "Valuation": valuation_score,
        "Dislocation": dislocation_score,
        "Quality": quality_score,
        "Growth": growth_score,
        "Momentum": momentum_score,
        "Penalty": penalty,
        "Score": opportunity,
        "Signal": signal,
        "Risk Flags": ", ".join(flags) if flags else "none",
        "Valuation Methods": methods,
    }

@st.cache_data(ttl=3600, show_spinner=False)
def run_scan():
    tickers = list(UNIVERSE.keys())
    prices = download_prices(tuple(tickers))
    rows = []
    failures = []
    if prices.empty:
        return pd.DataFrame(), ["Price download failed"]

    for symbol, company in UNIVERSE.items():
        try:
            if symbol not in prices.columns:
                failures.append(symbol)
                continue
            p = price_features(prices[symbol])
            info = get_info(symbol)
            row = score_stock(symbol, company, p, info)
            if row:
                rows.append(row)
            else:
                failures.append(symbol)
        except Exception:
            failures.append(symbol)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Score", "Discount"], ascending=[False, True]).reset_index(drop=True)
        df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df, failures

def pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"

def money(x):
    return "—" if pd.isna(x) else f"${x:,.2f}"

st.title("📉 Blue-Chip Value Scanner")
st.caption(
    "Find high-quality companies that may be temporarily trading below a reasonable value range."
)

with st.expander("How the score works"):
    st.markdown(
        """
**Opportunity Score (0–100)**

- **30% Valuation** — fair-value discount, P/E, PEG and free-cash-flow yield
- **25% Price dislocation** — 52-week drawdown and distance from the 200-day average
- **25% Business quality** — margins, ROE, debt and positive free cash flow
- **15% Growth** — recent revenue and earnings growth
- **5% Momentum** — small confirmation factor to avoid blindly buying falling knives

The model also applies **value-trap penalties** for signals such as sharply declining
revenue/earnings, excessive debt, negative margins or negative free cash flow.

**Fair value is a heuristic range, not a price target.** It blends a normalized earnings
multiple, free-cash-flow yield, and (when available) analyst consensus as a lower-weight
reference. Always verify company-specific news and filings before investing.
        """
    )

with st.spinner("Scanning 50 companies… first load can take a little while."):
    df, failures = run_scan()

if df.empty:
    st.error("No market data could be loaded. Try again later.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Stocks scanned", len(df))
c2.metric("Strong / Attractive", int((df["Score"] >= 70).sum()))
c3.metric("20%+ below fair value", int((df["Discount"] <= -0.20).sum()))
c4.metric("Median opportunity score", f'{df["Score"].median():.0f}')

st.subheader("Top opportunities")

min_score = st.slider("Minimum score", 0, 100, 60, 5)
only_discounted = st.toggle("Only show stocks below estimated fair value", value=True)

view = df[df["Score"] >= min_score].copy()
if only_discounted:
    view = view[view["Discount"] < 0]

display = view[
    ["Rank", "Ticker", "Company", "Price", "Fair Value", "Discount",
     "52W Drawdown", "Quality", "Growth", "Score", "Signal"]
].copy()
display["Discount"] = display["Discount"] * 100
display["52W Drawdown"] = display["52W Drawdown"] * 100

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn(format="$%.2f"),
        "Fair Value": st.column_config.NumberColumn(format="$%.2f"),
        "Discount": st.column_config.NumberColumn(format="%.1f%%"),
        "52W Drawdown": st.column_config.NumberColumn(format="%.1f%%"),
        "Quality": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        "Growth": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
        "Score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
    },
)

# A mobile-friendly detail section makes the raw numbers easier to interpret.
st.subheader("Stock detail")
selected = st.selectbox(
    "Choose a stock",
    options=view["Ticker"].tolist() if not view.empty else df["Ticker"].tolist(),
)
r = df.loc[df["Ticker"] == selected].iloc[0]

d1, d2, d3 = st.columns(3)
d1.metric(f"{r['Ticker']} price", money(r["Price"]))
d2.metric("Estimated fair value", money(r["Fair Value"]), pct(-r["Discount"]) + " upside vs price" if r["Discount"] < 0 else None)
d3.metric("Opportunity score", f"{r['Score']:.0f}/100")

st.markdown(
    f"""
**{r['Company']}** · {r['Sector']}  
**Estimated fair-value range:** {money(r['Fair Low'])} – {money(r['Fair High'])}  
**Price vs fair value:** {pct(r['Discount'])}  
**52-week drawdown:** {pct(r['52W Drawdown'])}  
**Revenue growth:** {pct(r['Revenue Growth'])} · **Earnings growth:** {pct(r['Earnings Growth'])}  
**ROE:** {pct(r['ROE'])} · **Profit margin:** {pct(r['Profit Margin'])}  
**Forward P/E:** {"—" if pd.isna(r['Forward P/E']) else f"{r['Forward P/E']:.1f}"}  
**FCF yield:** {pct(r['FCF Yield'])}  
**Risk flags:** {r['Risk Flags']}  
**Fair-value inputs used:** {r['Valuation Methods']}
    """
)

components = pd.DataFrame(
    {
        "Component": ["Valuation", "Price dislocation", "Quality", "Growth", "Momentum"],
        "Score": [r["Valuation"], r["Dislocation"], r["Quality"], r["Growth"], r["Momentum"]],
    }
).set_index("Component")
st.bar_chart(components)

st.caption(
    f"Data refreshed at approximately {datetime.now().strftime('%Y-%m-%d %H:%M')} on this server. "
    "Market data can be delayed or incomplete."
)

if failures:
    st.info("Some tickers could not be fully loaded: " + ", ".join(failures))

st.warning(
    "Educational/research tool only — not personalized investment advice. "
    "A low price can reflect real deterioration. Review earnings, filings, debt, competitive position, "
    "and recent company news before making an investment decision."
)
