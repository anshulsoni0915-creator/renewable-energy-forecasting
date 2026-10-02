**Live demo:** https://huggingface.co/spaces/AnshulSoni0915/renewable-energy-forecasting-app

# Renewable Energy Forecasting App

A stacked Bidirectional LSTM that forecasts solar, wind and grid energy output
for the next 24 hours from weather conditions, with an interactive Gradio interface.

## Model
- Two Bidirectional LSTM layers (128 and 64 units) with Dropout and BatchNormalization (TensorFlow/Keras)
- Input: 48 hours of history, 29 features (weather, time encodings, lag and rolling features)
- Output: 24 hourly forecasts for solar power, wind power and grid energy (MW)
- Features and targets scaled with MinMaxScaler (Scikit-Learn)
- Dataset: _name/source, number of rows, time range_

## Results
| Metric | Value |
|--------|-------|
| R²     | _your value_ |
| MAE    | _your value_ |
| RMSE   | _your value_ |

## Tech stack
Python, TensorFlow/Keras, Scikit-Learn, NumPy, Matplotlib, Gradio, Open-Meteo API

## Features
- **Live city forecast:** pulls the last 48 hours of real weather for any city (Open-Meteo) and forecasts the next 24 hours
- **Scenario lab:** presets (clear summer day, windy front, overcast winter, mild spring day) or your own weather
- **Uncertainty bands:** 10th to 90th percentile range using Monte Carlo Dropout
- **Plain-language advice:** best window for heavy loads or battery charging, steep solar-drop alerts, and a schedule-risk indicator
- **What drives the forecast:** a sensitivity table showing how cloud, sunlight, wind and temperature change the output

## Notes and limitations
- The model was trained on real 48-hour sequences from one dataset, so it is not calibrated for every location.
- The scenario lab approximates the past 48 hours from the weather you set (daylight curve for sunlight, other values constant).
- In the live tab, wind speed is measured at 10 m, and power features are estimated from weather with simple formulas.
- Weather data by [Open-Meteo.com](https://open-meteo.com).

## Run locally
```
pip install -r requirements.txt
python app.py
```

## Files
- `app.py`: Gradio interface and prediction logic
- `final_energy_lstm.h5`: trained model
- `feat_scaler.pkl`: fitted feature scaler (also used to convert outputs to MW)
