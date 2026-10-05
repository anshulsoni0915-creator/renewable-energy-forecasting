# ============================================================
# ⚡ Renewable Energy Dashboard — Gradio App
# Stacked Bidirectional LSTM: 48-hour lookback -> 24-hour forecast
#   • Live city forecast (real weather from Open-Meteo)
#   • Interactive charts with uncertainty bands (Monte Carlo Dropout)
#   • "What if?" energy scenarios and custom weather builder
# ============================================================

import os
import json
import datetime as dt
import urllib.parse
import urllib.request
import warnings

import numpy as np
import joblib
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ── Paths (work locally and inside the Hugging Face container) ──
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH       = os.path.join(BASE_DIR, "final_energy_lstm.h5")
FEAT_SCALER_PATH = os.path.join(BASE_DIR, "feat_scaler.pkl")

# ── Load model and feature scaler ───────────────────────────
model       = tf.keras.models.load_model(MODEL_PATH, compile=False)
feat_scaler = joblib.load(FEAT_SCALER_PATH)

# ── Constants ───────────────────────────────────────────────
LOOKBACK, HORIZON = 48, 24
HIST = LOOKBACK + 24             # 72 hours of history (24 extra feed the lag features)
CO2_T_PER_MWH = 0.7              # assumed grid emission factor (tonnes CO2 per MWh)
MC_SAMPLES    = 30               # Monte Carlo Dropout passes for the uncertainty band

# The model outputs 0-1 values scaled with the same min/max as these feature
# columns, so we convert back to MW with the feature scaler's own range.
IDX_SOLAR, IDX_WIND, IDX_GRID = 13, 14, 16

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
C_SOLAR, C_WIND, C_GRID, C_SHARE = "#fbbf24", "#60a5fa", "#fb7185", "#2dd4bf"


def to_mw(scaled, idx):
    lo, hi = feat_scaler.data_min_[idx], feat_scaler.data_max_[idx]
    return np.maximum(scaled * (hi - lo) + lo, 0)


# ── Feature engineering ─────────────────────────────────────
def _arr(v, n):
    return np.full(n, float(v)) if np.isscalar(v) else np.asarray(v, dtype=float)


def build_features(h):
    """Build the 48 x 29 model input from 72 hours of weather (dict of arrays)."""
    hrs = np.asarray(h["hrs"])
    n = len(hrs)
    temp, irr, wind, cloud, hum, pres, month, dow = (
        _arr(h[k], n) for k in ("temp", "irr", "wind", "cloud", "hum", "pres", "month", "dow"))

    density = 1.225 * (1 - 0.0065 * temp / 288.15) ** 5.255
    solar   = np.maximum(0, (irr / 1000) * 500 * (1 - 0.4 * cloud))
    wind_mw = np.where((wind >= 3) & (wind <= 25),
                       0.5 * density * 3.14 * 50**2 * wind**3 * 0.45 / 1e6, 0.0)
    hydro   = np.full(n, 90.0)
    grid    = solar + wind_mw + hydro

    def lag(x, k):  return np.concatenate([np.full(k, x[0]), x[:-k]])
    def roll(x, k): return np.array([x[max(0, i - k + 1): i + 1].mean() for i in range(len(x))])

    cols = [
        temp, irr, wind, cloud, hum, pres, density,
        np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        solar, wind_mw, hydro, grid,
        lag(solar, 1), lag(wind_mw, 1), lag(grid, 1),
        lag(solar, 24), lag(wind_mw, 24), lag(grid, 24),
        roll(solar, 24), roll(wind_mw, 24), roll(grid, 24),
        irr * temp, wind**3 * density, (solar + wind_mw) / (grid + 1),
    ]
    return np.stack(cols, axis=1)[-LOOKBACK:]


# ── Model calls ─────────────────────────────────────────────
def _scale(window):
    return np.clip(feat_scaler.transform(window), 0, 1)       # stay inside training range


def mc_dropout(X, samples=MC_SAMPLES):
    """Run the model with only the Dropout layers active (BatchNorm stays in inference mode)."""
    x = tf.convert_to_tensor(np.repeat(X, samples, axis=0).astype("float32"))
    for layer in model.layers:
        if type(layer).__name__ == "InputLayer":
            continue
        x = layer(x, training=(type(layer).__name__ == "Dropout"))
    return x.numpy()


def forecast_window(window):
    """Central forecast and P10-P90 band for solar, wind and grid (MW)."""
    X   = _scale(window).reshape(1, LOOKBACK, -1)
    det = model.predict(X, verbose=0)[0]
    mc  = mc_dropout(X)
    out = {}
    for key, i, idx in (("solar", 0, IDX_SOLAR), ("wind", 1, IDX_WIND), ("grid", 2, IDX_GRID)):
        out[key] = to_mw(det[:, i], idx)
        samples  = to_mw(mc[:, :, i], idx)
        out[key + "_lo"] = np.percentile(samples, 10, axis=0)
        out[key + "_hi"] = np.percentile(samples, 90, axis=0)
    return out


# ── Weather contexts: where the 72-hour history comes from ──
def custom_ctx(temp, irr_peak, wind, cloud, hum, pres, start_hour, month_name):
    """Custom scenario: sunlight follows a daylight curve, other weather is constant."""
    month, start = MONTHS.index(month_name) + 1, int(start_hour)
    n   = HIST + HORIZON
    hrs = (start - HIST + np.arange(n)) % 24
    irr = irr_peak * np.clip(np.sin(np.pi * (hrs - 6) / 12), 0, None)
    c   = lambda v: np.full(n, float(v))
    a = dict(temp=c(temp), irr=irr, wind=c(wind), cloud=c(cloud), hum=c(hum),
             pres=c(pres), hrs=hrs, month=c(month), dow=c(2))
    return {"hist": {k: v[:HIST] for k, v in a.items()},
            "fut":  {k: v[HIST:] for k, v in a.items()},
            "source": "scenario", "title": "Custom scenario"}


def _get_json(url):
    with urllib.request.urlopen(url, timeout=12) as r:
        return json.loads(r.read().decode("utf-8"))


def _clean(series):
    out, last = [], 0.0
    for v in series:
        last = last if v is None else float(v)
        out.append(last)
    return np.array(out)


def live_ctx(city):
    """Real weather from Open-Meteo (free for non-commercial use, no API key)."""
    geo = _get_json("https://geocoding-api.open-meteo.com/v1/search?"
                    + urllib.parse.urlencode({"name": city, "count": 1, "language": "en", "format": "json"}))
    if not geo.get("results"):
        raise ValueError(f"City '{city}' not found")
    g = geo["results"][0]
    w = _get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": g["latitude"], "longitude": g["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,"
                  "wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms", "past_days": 3, "forecast_days": 2, "timezone": "auto"}))
    h = w["hourly"]
    times = [dt.datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in h["time"]]
    now_local = (dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
                 + dt.timedelta(seconds=w.get("utc_offset_seconds", 0)))
    key = now_local.replace(minute=0, second=0, microsecond=0)
    idx = times.index(key) if key in times else 72 + now_local.hour
    if idx < HIST - 1 or idx + HORIZON >= len(times):
        raise ValueError("not enough weather data returned for this location")

    a = dict(temp=_clean(h["temperature_2m"]), irr=_clean(h["shortwave_radiation"]),
             wind=_clean(h["wind_speed_10m"]), cloud=_clean(h["cloud_cover"]) / 100,
             hum=_clean(h["relative_humidity_2m"]), pres=_clean(h["surface_pressure"]),
             hrs=np.array([t.hour for t in times]), month=np.array([t.month for t in times]),
             dow=np.array([t.weekday() for t in times]))
    place = ", ".join(p for p in (g.get("name"), g.get("admin1"), g.get("country")) if p)
    now = {k: a[k][idx] for k in ("temp", "cloud", "wind", "irr")}
    return {"hist": {k: v[idx - HIST + 1: idx + 1] for k, v in a.items()},
            "fut":  {k: v[idx + 1: idx + 1 + HORIZON] for k, v in a.items()},
            "source": "live", "title": place,
            "banner": (f"📍 **{place}** · local time {now_local:%H:%M} · now {now['temp']:.0f} °C, "
                       f"cloud {now['cloud'] * 100:.0f}%, wind {now['wind']:.1f} m/s, "
                       f"sunlight {now['irr']:.0f} W/m² · "
                       f"<sub>Weather data by [Open-Meteo.com](https://open-meteo.com)</sub>")}


# ── Charts (interactive Plotly) ─────────────────────────────
def _layout(fig, title, labels, height=340):
    step = max(1, len(labels) // 8)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eefc", size=12), height=height, margin=dict(l=8, r=8, t=56, b=8),
        title=dict(text=title, x=0.02, font=dict(size=17)), hovermode="x unified",
        legend=dict(orientation="h", y=1.14, x=1, xanchor="right", font=dict(size=10)))
    fig.update_xaxes(type="category", showgrid=False, tickmode="array",
                     tickvals=labels[::step], tickangle=0)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def _band(fig, x, lo, hi, rgba, name, **kw):
    fig.add_trace(go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False,
                             hoverinfo="skip"), **kw)
    fig.add_trace(go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=rgba, name=name, hoverinfo="skip"), **kw)


def solar_fig(f, labels, physics):
    fig = go.Figure()
    _band(fig, labels, f["solar_lo"], f["solar_hi"], "rgba(251,191,36,0.28)", "P10–P90 band")
    fig.add_trace(go.Scatter(x=labels, y=f["solar"], name="Solar (MW)", mode="lines",
                             line=dict(color=C_SOLAR, width=3), fill="tozeroy",
                             fillcolor="rgba(251,191,36,0.12)", hovertemplate="%{y:.0f} MW"))
    if physics is not None:
        fig.add_trace(go.Scatter(x=labels, y=physics, name="Simple physics (weather forecast)",
                                 mode="lines", line=dict(color="white", width=1.3, dash="dash"),
                                 hovertemplate="%{y:.0f} MW"))
    pk = int(np.argmax(f["solar"]))
    fig.add_trace(go.Scatter(x=[labels[pk]], y=[f["solar"][pk]], mode="markers+text", showlegend=False,
                             text=[f"Peak {f['solar'][pk]:.0f} MW"], textposition="top center",
                             marker=dict(size=11, color="white", line=dict(color=C_SOLAR, width=3)),
                             hoverinfo="skip"))
    fig.update_yaxes(title_text="MW", range=[0, max(f["solar_hi"].max(), 1) * 1.25])
    return _layout(fig, "☀️ Solar Forecast", labels)


def wind_fig(f, labels, speed):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    _band(fig, labels, f["wind_lo"], f["wind_hi"], "rgba(96,165,250,0.28)", "P10–P90 band",
          secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=f["wind"], name="Wind output (MW)", mode="lines+markers",
                             line=dict(color=C_WIND, width=3), marker=dict(size=5),
                             hovertemplate="%{y:.1f} MW"), secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=speed, name="Wind speed (m/s)", mode="lines",
                             line=dict(color="#67e8f9", width=1.6, dash="dot"),
                             hovertemplate="%{y:.1f} m/s"), secondary_y=True)
    fig.update_yaxes(title_text="MW", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="m/s", secondary_y=True, showgrid=False, rangemode="tozero")
    return _layout(fig, "💨 Wind Prediction", labels)


def grid_fig(f, labels, share):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=share, name="Solar + wind share (%)", marker_color=C_SHARE,
                         opacity=0.35, hovertemplate="%{y:.0f}%"), secondary_y=True)
    _band(fig, labels, f["grid_lo"], f["grid_hi"], "rgba(251,113,133,0.25)", "P10–P90 band",
          secondary_y=False)
    fig.add_trace(go.Scatter(x=labels, y=f["grid"], name="Grid energy (MW)", mode="lines+markers",
                             line=dict(color=C_GRID, width=3), marker=dict(size=5),
                             hovertemplate="%{y:.0f} MW"), secondary_y=False)
    fig.update_yaxes(title_text="MW", secondary_y=False, rangemode="tozero")
    fig.update_yaxes(title_text="share %", secondary_y=True, showgrid=False, range=[0, 100])
    return _layout(fig, "⚡ Grid Energy & Renewable Share", labels)


# ── KPI cards and advice ────────────────────────────────────
def make_kpis(f, labels, share, speed):
    solar_mwh, wind_mwh = f["solar"].sum(), f["wind"].sum()          # hourly MW summed = MWh
    co2 = (solar_mwh + wind_mwh) * CO2_T_PER_MWH
    pk, best = int(np.argmax(f["solar"])), int(np.argmax(share))
    sp = float(np.mean(speed))
    wind_state = "Light winds" if sp < 3 else "Moderate winds" if sp < 8 else "Strong winds"

    def card(cls, icon, title, value, unit, chip):
        return (f"<div class='kpi {cls}'><div class='kpi-head'><span class='ico'>{icon}</span>{title}</div>"
                f"<div class='kpi-val'>{value}<small> {unit}</small></div><div class='chip'>{chip}</div></div>")

    return ("<div class='kpis'>"
            + card("k-solar", "☀️", "Solar Output", f"{f['solar'].max():.0f}", "MW peak", f"at {labels[pk]}")
            + card("k-wind", "💨", "Wind Power", f"{f['wind'].mean():.1f}", "MW avg", wind_state)
            + card("k-grid", "⚡", "Grid Energy", f"{f['grid'].max():.0f}", "MW peak", f"avg {f['grid'].mean():.0f} MW")
            + card("k-share", "🌱", "Renewable Share", f"{share.mean():.0f}", "% avg", f"best at {labels[best]}")
            + card("k-co2", "🌍", "CO₂ Avoided (est.)", f"{co2:,.0f}", "t", f"{solar_mwh + wind_mwh:,.0f} MWh clean energy")
            + "</div>")


def make_insights(f, labels, clock):
    solar, lo, hi = f["solar"], f["solar_lo"], f["solar_hi"]
    lines = ["### 💡 What to do with this forecast"]
    if solar.max() > 1:
        sums = np.convolve(solar, np.ones(3), "valid")
        i = int(np.argmax(sums))
        lines.append(f"- **Best 3-hour window for heavy loads or battery charging:** "
                     f"{labels[i]}–{(clock[i] + 3) % 24:02d}:00 (average solar {sums[i] / 3:.0f} MW).")
        drops = solar[:-1] - solar[1:]
        j = int(np.argmax(drops))
        if drops[j] > 0.12 * solar.max():
            lines.append(f"- ⚠️ **Steep solar drop:** about {drops[j]:.0f} MW lost between "
                         f"{labels[j]} and {labels[j + 1]}. Other sources or storage must cover the gap.")
        day = solar > 0.1 * solar.max()
        if day.sum() > 0:
            rel = ((hi - lo) / 2 / np.maximum(solar, 1e-6))[day] * 100
            lines.append(f"- 📏 **Schedule-risk indicator:** in daylight hours the model's uncertainty band "
                         f"averages ±{rel.mean():.1f}% of the solar forecast; {int((rel > 5).sum())} of "
                         f"{int(day.sum())} hours exceed ±5% and {int((rel > 10).sum())} exceed ±10%. "
                         f"India's tolerance band for solar is being tightened toward ±5%, so wide-band hours "
                         f"are harder to schedule. *(A model-uncertainty indicator, not a regulatory calculation.)*")
    else:
        lines.append("- Very little solar is expected in this window.")
    return "\n".join(lines)


# ── Dashboard render ────────────────────────────────────────
def render(ctx, banner):
    f = forecast_window(build_features(ctx["hist"]))
    clock  = [int(x) for x in ctx["fut"]["hrs"]]
    labels = [f"{x:02d}:00" for x in clock]
    share  = np.clip((f["solar"] + f["wind"]) / np.maximum(f["grid"], 1e-6) * 100, 0, 100)
    fut = ctx["fut"]
    physics = (np.maximum(0, (fut["irr"] / 1000) * 500 * (1 - 0.4 * fut["cloud"]))
               if ctx["source"] == "live" else None)
    return (ctx, banner, make_kpis(f, labels, share, fut["wind"]),
            solar_fig(f, labels, physics), wind_fig(f, labels, fut["wind"]),
            grid_fig(f, labels, share), make_insights(f, labels, clock))


def load_live(city):
    city = (city or "").strip() or "Delhi"
    try:
        ctx = live_ctx(city)
        return render(ctx, ctx["banner"])
    except Exception as e:                               # offline, unknown city, ...
        ctx = custom_ctx(*PRESETS["🌤️ Mild spring day"])
        return render(ctx, f"⚠️ Live weather unavailable ({e}). Showing a demo scenario instead. "
                           f"Check the city name or try again.")


def load_custom(*params):
    ctx = custom_ctx(*params)
    return render(ctx, "🎛️ **Custom scenario** — the past 48 hours are approximated from the weather you set "
                       "(sunlight follows a daylight curve, other values constant).")


# ── "What if?" energy scenarios ─────────────────────────────
def run_whatif(ctx, var, pct):
    if not ctx:
        return "<div class='whatif-res'>Run a forecast first.</div>", None
    p = float(pct) / 100
    h = {k: np.array(v, copy=True) for k, v in ctx["hist"].items()}
    if var == "Wind speed":
        h["wind"] = np.maximum(h["wind"] * (1 + p), 0)
    elif var == "Sunlight":
        h["irr"] = np.maximum(h["irr"] * (1 + p), 0)
    else:                                                    # cloud cover, in percentage points
        h["cloud"] = np.clip(h["cloud"] + p, 0, 1)

    X = np.stack([_scale(build_features(ctx["hist"])), _scale(build_features(h))])
    pr = model.predict(X, verbose=0)
    base, new = ({"solar": to_mw(q[:, 0], IDX_SOLAR), "wind": to_mw(q[:, 1], IDX_WIND),
                  "grid": to_mw(q[:, 2], IDX_GRID)} for q in pr)
    labels = [f"{int(x):02d}:00" for x in ctx["fut"]["hrs"]]

    def pct_change(a, b):
        return "—" if b < 1e-6 else f"{(a - b) / b * 100:+.1f}%"

    sh = lambda d: np.mean(np.clip((d["solar"] + d["wind"]) / np.maximum(d["grid"], 1e-6) * 100, 0, 100))
    unit = "points" if var == "Cloud cover" else "%"
    cards = (f"<div class='whatif-res'><div class='whatif-q'>What if <b>{var.lower()}</b> changes by "
             f"<b>{pct:+.0f}{unit if unit == '%' else ' points'}</b>?</div><div class='kpis small'>"
             f"<div class='kpi k-solar'><div class='kpi-head'>☀️ Solar energy</div><div class='kpi-val'>"
             f"{pct_change(new['solar'].sum(), base['solar'].sum())}</div></div>"
             f"<div class='kpi k-wind'><div class='kpi-head'>💨 Wind energy</div><div class='kpi-val'>"
             f"{pct_change(new['wind'].sum(), base['wind'].sum())}</div></div>"
             f"<div class='kpi k-grid'><div class='kpi-head'>⚡ Grid energy</div><div class='kpi-val'>"
             f"{pct_change(new['grid'].sum(), base['grid'].sum())}</div></div>"
             f"<div class='kpi k-share'><div class='kpi-head'>🌱 Renewable share</div><div class='kpi-val'>"
             f"{sh(new) - sh(base):+.1f}<small> pts</small></div></div>"
             f"<div class='note'>The model learns patterns from data, so small changes in outputs you did not "
             f"touch (for example solar when wind changes) reflect shared features, not physical cause and effect."
             f"</div></div>")

    fig = go.Figure()
    for key, color, name in (("solar", C_SOLAR, "Solar"), ("wind", C_WIND, "Wind"), ("grid", C_GRID, "Grid")):
        fig.add_trace(go.Scatter(x=labels, y=base[key], name=f"{name} — baseline", mode="lines",
                                 line=dict(color=color, width=1.6, dash="dash"), opacity=0.6,
                                 hovertemplate="%{y:.0f} MW"))
        fig.add_trace(go.Scatter(x=labels, y=new[key], name=f"{name} — scenario", mode="lines",
                                 line=dict(color=color, width=3), hovertemplate="%{y:.0f} MW"))
    fig.update_yaxes(title_text="MW", rangemode="tozero")
    return cards, _layout(fig, "Baseline vs scenario", labels, height=360)


# ── Presets for the custom scenario builder ─────────────────
# (temp, peak irradiance, wind, cloud, humidity, pressure, start hour, month)
PRESETS = {
    "☀️ Clear summer day": (33, 950, 5,  0.05, 35, 1008, 6, "Jun"),
    "🌬️ Windy front":      (16, 400, 16, 0.60, 70, 1002, 0, "Sep"),
    "☁️ Overcast winter":  (8,  250, 4,  0.85, 80, 1022, 6, "Jan"),
    "🌤️ Mild spring day":  (22, 650, 8,  0.30, 55, 1013, 6, "Apr"),
}

# ── Styling ─────────────────────────────────────────────────
CSS = """
html, body, .gradio-container, .gradio-container.dark {
  background: radial-gradient(1100px 500px at 8% -10%, rgba(56,189,248,.20), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(99,102,241,.22), transparent 60%),
              linear-gradient(160deg,#050b1f 0%,#0a1b46 55%,#0b2a5c 100%) !important;
  background-attachment: fixed !important; color:#e8eefc !important; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; width: 100% !important; }
html, body { min-height: 100vh; }
.block, .form, .gr-group, .plot-container, .gradio-container .panel {
  background: transparent !important; border: none !important; box-shadow: none !important; }
.glass, .glass.block, .glass > .block {
  background: rgba(255,255,255,.06) !important; border: 1px solid rgba(255,255,255,.13) !important;
  border-radius: 18px !important; backdrop-filter: blur(10px); padding: 8px !important; }
.brand { display:flex; align-items:center; gap:14px; padding: 8px 4px 2px; }
.brand .logo { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center;
  font-size:26px; background: linear-gradient(135deg,#38bdf8,#6366f1); box-shadow:0 6px 20px rgba(56,189,248,.35); }
.brand h1 { margin:0; font-size: 28px; font-weight:800; letter-spacing:.2px; color:#fff; }
.brand p { margin:0; opacity:.65; font-size:13px; }
.topbar { align-items:center !important; gap:10px !important; }
.topbar input { background: rgba(255,255,255,.08) !important; border:1px solid rgba(255,255,255,.18) !important;
  border-radius: 999px !important; color:#fff !important; padding: 12px 18px !important; font-size:15px !important; }
button.primary, .primary { background: linear-gradient(135deg,#2563eb,#4f46e5) !important; border:none !important;
  color:#fff !important; border-radius: 12px !important; font-weight:700 !important;
  box-shadow: 0 6px 18px rgba(79,70,229,.4) !important; }
button.primary:hover { filter: brightness(1.12); transform: translateY(-1px); }
button.secondary { background: rgba(255,255,255,.08) !important; border:1px solid rgba(255,255,255,.18) !important;
  color:#e8eefc !important; border-radius: 12px !important; }
button[role="tab"] { color:#aab8dd !important; font-weight:600 !important; font-size:15px !important;
  border:none !important; background:transparent !important; }
button[role="tab"][aria-selected="true"] { color:#fff !important; border-bottom: 3px solid #38bdf8 !important; }
.tab-wrapper, .tab-container { border-color: rgba(255,255,255,.12) !important; }
.welcome { font-size: 30px; font-weight: 800; color:#fff; margin: 14px 0 6px; }
.welcome span { font-size: 17px; font-weight: 400; opacity:.75; margin-left:8px; }
.banner { font-size: 13.5px; opacity:.9; padding: 2px 6px; }
.banner * { color:#cfe0ff !important; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:14px; margin: 8px 0 14px; }
.kpis.small { grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); }
.kpi { border-radius:18px; padding:16px 18px; border:1px solid rgba(255,255,255,.16); position:relative;
  backdrop-filter: blur(10px); transition: transform .15s ease, box-shadow .15s ease; }
.kpi:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(0,0,0,.35); }
.k-solar { background: linear-gradient(145deg, rgba(251,191,36,.30), rgba(251,146,60,.10)); }
.k-wind  { background: linear-gradient(145deg, rgba(96,165,250,.30), rgba(99,102,241,.10)); }
.k-grid  { background: linear-gradient(145deg, rgba(251,113,133,.28), rgba(244,63,94,.08)); }
.k-share { background: linear-gradient(145deg, rgba(45,212,191,.28), rgba(16,185,129,.08)); }
.k-co2   { background: linear-gradient(145deg, rgba(167,139,250,.28), rgba(139,92,246,.08)); }
.kpi-head { font-size:14px; font-weight:700; color:#fff; display:flex; align-items:center; gap:8px; }
.kpi .ico { font-size:22px; }
.kpi-val { font-size:38px; font-weight:800; color:#fff; line-height:1.15; margin-top:6px; }
.kpi-val small { font-size:15px; font-weight:600; opacity:.8; }
.chip { display:inline-block; margin-top:8px; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700;
  background: rgba(255,255,255,.16); color:#fff; }
.sec-title { font-size:22px; font-weight:800; color:#fff; margin: 18px 4px 6px; }
.sec-title span { font-size:13px; font-weight:400; opacity:.65; margin-left:10px; }
.whatif-q { font-size:20px; color:#fff; margin: 6px 4px 10px; }
.whatif-q b { color:#fde68a; }
.prose, .prose *, .md, .md * { color:#e8eefc !important; }
.glass h3 { color:#fff !important; }
label, .block label span, span.svelte-1gfkn6j { color:#c7d4f5 !important; }
input, select, textarea { background: rgba(255,255,255,.08) !important; color:#fff !important;
  border:1px solid rgba(255,255,255,.18) !important; }
.note { font-size:12px; opacity:.6; margin: 0 4px 6px; }
.foot { text-align:center; opacity:.55; font-size:12px; margin: 18px 0 6px; }
"""
JS = "() => { document.body.classList.add('dark'); }"

BRAND = ("<div class='brand'><div class='logo'>⚡</div><div><h1>Renewable Energy Dashboard</h1>"
         "<p>Stacked Bidirectional LSTM forecasts for solar, wind and grid energy</p></div></div>")

# ── Interface ───────────────────────────────────────────────
with gr.Blocks(title="Renewable Energy Dashboard") as demo:
    ctx_state = gr.State(None)
    gr.HTML(BRAND)
    with gr.Row(elem_classes="topbar"):
        city = gr.Textbox(value="Delhi", show_label=False, container=False, scale=5,
                          placeholder="🔍  Search a city, e.g. Jaipur, Chennai, London")
        go_btn = gr.Button("Get forecast", variant="primary", scale=1, min_width=150)
    banner = gr.Markdown(elem_classes="banner")

    with gr.Tabs():
        with gr.Tab("📊 Dashboard"):
            gr.HTML("<div class='welcome'>Welcome!<span>Get the latest renewable energy outlook.</span></div>")
            kpis = gr.HTML()
            with gr.Row():
                solar_plot = gr.Plot(show_label=False, elem_classes="glass")
                wind_plot  = gr.Plot(show_label=False, elem_classes="glass")
            with gr.Row():
                grid_plot = gr.Plot(show_label=False, elem_classes="glass", scale=3)
                advice    = gr.Markdown(elem_classes="glass", container=True)

            gr.HTML("<div class='sec-title'>⚡ Energy Scenarios"
                    "<span>Test how the forecast reacts if the weather changes</span></div>")
            with gr.Row(elem_classes="glass"):
                wi_var = gr.Dropdown(["Wind speed", "Sunlight", "Cloud cover"], value="Wind speed",
                                     label="What if this changes…")
                wi_pct = gr.Slider(-50, 50, value=-25, step=5, label="…by (%, or points for cloud cover)")
                wi_btn = gr.Button("▶ Run simulation", variant="primary")
            wi_cards = gr.HTML()
            wi_plot  = gr.Plot(show_label=False, elem_classes="glass")

            with gr.Accordion("🎛️ Build your own weather (custom scenario)", open=False, elem_classes="glass"):
                with gr.Row():
                    preset_btns = {n: gr.Button(n, variant="secondary") for n in PRESETS}
                with gr.Row():
                    with gr.Column():
                        temp  = gr.Slider(-10, 45,   value=22,   label="🌡️ Temperature (°C)")
                        irr   = gr.Slider(0,   1000, value=650,  label="☀️ Peak midday solar irradiance (W/m²)")
                        wind  = gr.Slider(0,   30,   value=8,    label="💨 Wind speed (m/s)")
                        cloud = gr.Slider(0,   1,    value=0.3,  step=0.05, label="☁️ Cloud cover (0-1)")
                    with gr.Column():
                        hum   = gr.Slider(10,  100,  value=55,   label="💧 Humidity (%)")
                        pres  = gr.Slider(990, 1030, value=1013, label="🔵 Air pressure (hPa)")
                        start = gr.Slider(0,   23,   value=6,    step=1, label="🕒 Forecast starts at (hour of day)")
                        month = gr.Dropdown(MONTHS, value="Apr", label="📅 Month")
                run_btn = gr.Button("🔮 Run custom scenario", variant="primary")

        with gr.Tab("🧠 How it works"):
            gr.Markdown(
                "### Model\n"
                "- Two **Bidirectional LSTM** layers (128 and 64 units) with Dropout and BatchNormalization\n"
                "- **Input:** 48 hours of history x 29 features (weather, time-of-day / month encodings, lag and rolling-average features)\n"
                "- **Output:** 24 hourly values each for solar power, wind power and grid energy\n\n"
                "### Uncertainty bands\n"
                f"The shaded bands show the 10th to 90th percentile of {MC_SAMPLES} forecasts made with "
                "**Monte Carlo Dropout** (Dropout kept on at prediction time). Wide bands mean the model is less sure.\n\n"
                "### Live city forecast\n"
                "Real hourly weather for the last 48 hours comes from the free Open-Meteo service. The model reads it "
                "and forecasts the next 24 hours. The dashed line is a simple physics estimate from the weather "
                "service's own forecast, shown for comparison (uncalibrated). Wind speed is measured at 10 m, not "
                "turbine height, and power features are estimated with simple formulas, so treat results as illustrative.\n\n"
                "### Energy scenarios\n"
                "The simulator changes one weather input across the whole 48-hour history (for example wind speed "
                "−25%) and re-runs the model, so you can see how sensitive the forecast is.\n\n"
                "### Custom scenario\n"
                "With no live data feed, the builder **approximates the past 48 hours from the weather you set**: "
                "sunlight follows a daylight curve and other weather is held constant.\n\n"
                "### Assumptions\n"
                f"- CO₂ avoided uses a rough grid emission factor of {CO2_T_PER_MWH} t/MWh.\n"
                "- Hydro is treated as a constant base load.\n"
                "- The model was trained on one dataset, so it is not calibrated for every location.\n",
                elem_classes="glass")
    gr.HTML("<div class='foot'>Weather data by Open-Meteo.com · Built with TensorFlow, Gradio and Plotly</div>")

    # ── Events ──
    outs     = [ctx_state, banner, kpis, solar_plot, wind_plot, grid_plot, advice]
    wi_in    = [ctx_state, wi_var, wi_pct]
    wi_out   = [wi_cards, wi_plot]
    custom_in = [temp, irr, wind, cloud, hum, pres, start, month]

    go_btn.click(load_live, city, outs).then(run_whatif, wi_in, wi_out)
    city.submit(load_live, city, outs).then(run_whatif, wi_in, wi_out)
    demo.load(load_live, city, outs).then(run_whatif, wi_in, wi_out)
    wi_btn.click(run_whatif, wi_in, wi_out)
    run_btn.click(load_custom, custom_in, outs).then(run_whatif, wi_in, wi_out)
    for name, btn in preset_btns.items():
        btn.click(lambda p=PRESETS[name]: list(p), None, custom_in).then(
            load_custom, custom_in, outs).then(run_whatif, wi_in, wi_out)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"), css=CSS, js=JS)
