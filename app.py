# ============================================================
# Renewable Energy Forecasting — Gradio App for HuggingFace
# ============================================================

import os
import numpy as np
import joblib
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# ── Resolve Paths (works both locally and inside HuggingFace container) ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH      = os.path.join(BASE_DIR, 'final_energy_lstm.h5')
FEAT_SCALER_PATH = os.path.join(BASE_DIR, 'feat_scaler.pkl')
TGT_SCALER_PATH  = os.path.join(BASE_DIR, 'tgt_scaler.pkl')

# ── Load Model & Scalers ─────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
feat_scaler = joblib.load(FEAT_SCALER_PATH)
tgt_scaler  = joblib.load(TGT_SCALER_PATH)
print("✅ Model and scalers loaded successfully!")

# ── Constants ────────────────────────────────────────────────
LOOKBACK = 48
HORIZON  = 24
N_TARGETS = 3

INPUT_FEATURES = [
    "temperature", "solar_irradiance", "wind_speed", "cloud_cover",
    "humidity", "air_pressure", "air_density",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "solar_power_mw", "wind_power_mw", "hydro_power_mw", "grid_energy_mw",
    "solar_power_mw_lag1", "wind_power_mw_lag1", "grid_energy_mw_lag1",
    "solar_power_mw_lag24", "wind_power_mw_lag24", "grid_energy_mw_lag24",
    "solar_power_mw_roll24_mean", "wind_power_mw_roll24_mean", "grid_energy_mw_roll24_mean",
    "irr_temp_interaction", "wind_density_power", "renewable_fraction"
]

# ── Forecast Function ─────────────────────────────────────────
def forecast_energy(temperature, solar_irradiance, wind_speed, cloud_cover,
                    humidity, air_pressure, horizon_choice):

    horizon_map = {
        "Next 24 Hours (Hourly)":   24,
        "Next 7 Days (Daily)":       7,
        "Next 12 Months (Monthly)": 12
    }
    n_steps = horizon_map[horizon_choice]

    # Derived weather values
    air_density_val = 1.225 * (1 - 0.0065 * temperature / 288.15) ** 5.255
    hour_sin  = np.sin(2 * np.pi * 12 / 24)
    hour_cos  = np.cos(2 * np.pi * 12 / 24)
    month_sin = np.sin(2 * np.pi * 6  / 12)
    month_cos = np.cos(2 * np.pi * 6  / 12)
    dow_sin   = np.sin(2 * np.pi * 2  / 7)
    dow_cos   = np.cos(2 * np.pi * 2  / 7)

    # Energy estimates from weather inputs
    solar_est = max(0, (solar_irradiance / 1000) * 500 * (1 - 0.4 * cloud_cover))
    wind_est  = (max(0, 0.5 * air_density_val * 3.14 * (50**2) * (wind_speed**3) * 0.45 / 1e6)
                 if 3 <= wind_speed <= 25 else 0)
    hydro_est = 90
    grid_est  = solar_est + wind_est + hydro_est

    # Build input row matching INPUT_FEATURES order
    base_row = np.array([
        temperature, solar_irradiance, wind_speed, cloud_cover,
        humidity, air_pressure, air_density_val,
        hour_sin, hour_cos, month_sin, month_cos, dow_sin, dow_cos,
        solar_est, wind_est, hydro_est, grid_est,
        solar_est, wind_est, grid_est,   # lag1
        solar_est, wind_est, grid_est,   # lag24
        solar_est, wind_est, grid_est,   # roll24_mean
        solar_irradiance * temperature,  # irr_temp_interaction
        wind_speed**3 * air_density_val, # wind_density_power
        (solar_est + wind_est) / (grid_est + 1)  # renewable_fraction
    ])

    # Build lookback window with small noise
    noise         = np.random.normal(0, 0.02, (LOOKBACK, len(base_row)))
    window        = np.tile(base_row, (LOOKBACK, 1)) + noise
    window_scaled = feat_scaler.transform(window)
    X_input       = window_scaled.reshape(1, LOOKBACK, -1)

    # Predict
    pred_scaled = model.predict(X_input, verbose=0)[0]
    pred_inv    = tgt_scaler.inverse_transform(pred_scaled)
    pred_inv    = np.maximum(pred_inv, 0)

    steps_to_use = min(n_steps, HORIZON)
    solar_fc = pred_inv[:steps_to_use, 0]
    wind_fc  = pred_inv[:steps_to_use, 1]
    grid_fc  = pred_inv[:steps_to_use, 2]

    xlabel_map = {
        "Next 24 Hours (Hourly)":   "Hour",
        "Next 7 Days (Daily)":      "Day",
        "Next 12 Months (Monthly)": "Month"
    }
    xvals  = np.arange(1, steps_to_use + 1)
    xlabel = xlabel_map[horizon_choice]

    # ── Plot ─────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    fig.patch.set_facecolor("#0f1117")

    for ax, vals, color, label in zip(
        axes,
        [solar_fc, wind_fc, grid_fc],
        ["#F4A261", "#457B9D", "#E76F51"],
        ["☀️ Solar Power (MW)", "💨 Wind Power (MW)", "⚡ Grid Energy (MW)"]
    ):
        ax.set_facecolor("#1a1c24")
        ax.plot(xvals, vals, "-o", color=color, linewidth=2, markersize=5)
        ax.fill_between(xvals, vals, alpha=0.2, color=color)
        ax.set_ylabel(label, fontsize=10, color="white", fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#555")
        avg = vals.mean()
        ax.axhline(avg, color=color, linestyle="--", alpha=0.5, linewidth=1)
        ax.text(xvals[-1], avg, f"  avg:{avg:.1f}", color=color, fontsize=8, va="center")

    axes[-1].set_xlabel(xlabel, color="white", fontsize=11)
    axes[-1].tick_params(axis="x", colors="white")
    plt.suptitle(f"⚡ Renewable Energy Forecast — {horizon_choice}",
                 fontsize=13, fontweight="bold", color="white")
    plt.tight_layout()

    # ── Summary Text ─────────────────────────────────────────
    summary = (
        f"📊 Forecast Summary\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"  ☀️  Solar  — Avg: {solar_fc.mean():.1f} MW | Peak: {solar_fc.max():.1f} MW\n"
        f"  💨  Wind   — Avg: {wind_fc.mean():.1f} MW  | Peak: {wind_fc.max():.1f} MW\n"
        f"  ⚡  Grid   — Avg: {grid_fc.mean():.1f} MW | Peak: {grid_fc.max():.1f} MW\n"
        f"  🌱  Renewable Share: {((solar_fc + wind_fc) / grid_fc * 100).mean():.1f}%"
    )
    return fig, summary


# ── Gradio Interface ──────────────────────────────────────────
demo = gr.Interface(
    fn=forecast_energy,
    inputs=[
        gr.Slider(-10, 45,   value=22,   label="🌡️ Temperature (°C)"),
        gr.Slider(0,   1000, value=650,  label="☀️ Solar Irradiance (W/m²)"),
        gr.Slider(0,   30,   value=8,    label="💨 Wind Speed (m/s)"),
        gr.Slider(0,   1,    value=0.3,  step=0.05, label="☁️ Cloud Cover (0-1)"),
        gr.Slider(10,  100,  value=55,   label="💧 Humidity (%)"),
        gr.Slider(990, 1030, value=1013, label="🔵 Air Pressure (hPa)"),
        gr.Dropdown(
            choices=["Next 24 Hours (Hourly)", "Next 7 Days (Daily)", "Next 12 Months (Monthly)"],
            value="Next 24 Hours (Hourly)",
            label="📅 Forecasting Horizon"
        )
    ],
    outputs=[
        gr.Plot(label="📈 Energy Forecast Chart"),
        gr.Textbox(label="📊 Forecast Summary", lines=7)
    ],
    title="⚡ Renewable Energy Generation Forecasting System",
    description="🤖 AI-powered Bidirectional LSTM forecast for Solar, Wind & Grid energy. Adjust weather conditions and choose your forecasting horizon.",
    theme=gr.themes.Soft(),
    examples=[
        [25, 750, 10, 0.2, 45, 1015, "Next 24 Hours (Hourly)"],
        [5,  200, 15, 0.7, 80, 1005, "Next 7 Days (Daily)"],
        [18, 500, 7,  0.4, 60, 1012, "Next 12 Months (Monthly)"],
    ]
)

if __name__ == "__main__":
    demo.launch()
