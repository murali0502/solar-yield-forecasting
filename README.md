# ☀️ Solar Yield Forecasting

<p align="center">
  <strong>From environmental signals to intelligent solar yield forecasts.</strong>
</p>

<p align="center">
  An end-to-end hybrid machine learning platform combining
  <strong>XGBoost</strong> and <strong>Temporal Convolutional Networks (TCN)</strong>
  for solar output forecasting.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-189FDD)
![TensorFlow](https://img.shields.io/badge/TensorFlow-TCN-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-Frontend-61DAFB?logo=react&logoColor=black)
![TypeScript](https://img.shields.io/badge/TypeScript-Frontend-3178C6?logo=typescript&logoColor=white)

</p>

---

## ⚡ Overview

Solar generation is influenced by continuously changing environmental
and temporal conditions.

Irradiance, cloud cover, temperature, humidity, wind conditions,
and time-dependent patterns can all affect solar output.

This project addresses the forecasting problem through a **hybrid
machine learning architecture** that combines two complementary
models:

- 🌳 **XGBoost** — learns structured feature relationships
- 🌊 **TCN** — learns temporal patterns from historical sequences
- 🔗 **Hybrid Ensemble** — combines both model predictions

The forecasting pipeline is exposed through a **FastAPI REST API**
and connected to an interactive **React + TypeScript dashboard**.

```text
Environmental + Temporal Data
              │
              ▼
       Data Preprocessing
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
   XGBoost          TCN
       │             │
       └──────┬──────┘
              ▼
      Hybrid Ensemble
              │
              ▼
     Solar Yield Forecast
              │
              ▼
         FastAPI API
              │
              ▼
       React Dashboard
