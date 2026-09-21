# ☀️ Solar Yield Forecasting

### Turning environmental signals into actionable solar forecasts.

A hybrid machine learning forecasting system combining **XGBoost** and
**Temporal Convolutional Networks (TCN)** to model both structured
environmental relationships and temporal patterns in solar generation.

<p align="center">

![Solar Forecasting](frontend/src/assets/hero.png)

</p>

<p align="center">

`Python` · `XGBoost` · `TCN` · `TensorFlow` · `FastAPI` · `React`

</p>

## ⚡ Why Hybrid Forecasting?

Solar generation is not governed by a single signal.

Cloud cover changes rapidly.
Irradiance varies throughout the day.
Temperature and humidity affect generation.
Time introduces strong periodic patterns.

A single model may capture some of these relationships.

This system approaches the problem differently:

> **Use different models for different types of information,
> then combine their forecasts.**

## 🧠 Inside the System

| Layer | Responsibility |
|---|---|
| Data Layer | Environmental + temporal observations |
| Preprocessing | Feature preparation and scaling |
| XGBoost | Structured feature learning |
| TCN | Temporal sequence modelling |
| Hybrid Engine | Combines model predictions |
| FastAPI | Model inference API |
| React UI | Interactive forecasting dashboard |

## 🧩 Two Models. One Forecast.

### 🌳 XGBoost
Designed for structured feature relationships.

**Understands:**

`Temperature` · `Humidity` · `Cloud Cover` · `Wind Speed`
· `Irradiance` · `Time Features`

### 🌊 TCN
Designed for temporal behaviour.

**Looks back across:**

`24 hourly observations`

to identify temporal patterns that aren't represented by a single
observation.

### ☀️ Hybrid Ensemble

The final forecast combines the complementary predictions of both models.

```text
XGBoost ───────┐
               ├──► Hybrid Ensemble ──► Solar Yield
TCN ───────────┘


---

5. Add a "System in 30 seconds"

This is very useful for clients/recruiters.

```markdown
## 🚀 System in 30 Seconds

```text
INPUT
  │
  ├── Weather conditions
  ├── Irradiance
  └── Temporal information
          │
          ▼
     PREPROCESSING
          │
     ┌────┴────┐
     ▼         ▼
 XGBoost      TCN
     │         │
     └────┬────┘
          ▼
   HYBRID ENSEMBLE
          │
          ▼
    SOLAR YIELD
     FORECAST


---

6. Showcase the actual application

Don't bury the frontend.

```markdown
## 🖥️ Forecasting Dashboard

The project includes an interactive web interface for submitting
forecast inputs and visualizing predicted solar yield.

<p align="center">

<img src="..." width="900">

</p>

### The application provides

- Forecast generation
- Model-driven predictions
- Forecast visualization
- API-backed inference
- Model performance information

## 📊 Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | — | — | — |
| TCN | — | — | — |
| Hybrid | — | — | — |

## 🛠️ Quick Start

### 1. Clone

```bash
git clone https://github.com/murali0502/solar-yield-forecasting.git
cd solar-yield-forecasting

### 2. Backend

python -m venv .venv

Windows:
.venv\Scripts\activate

Install Dependencies:
pip install -r requirements.txt

Run:
python -m uvicorn api.main:app --reload

### 3. Frontend
cd frontend
npm install
npm run dev


---

9. Add an architecture diagram

This is where I'd make your README **different from ordinary GitHub projects**.

Instead of only Markdown, create a proper architecture image:

```text
              ┌───────────────────┐
              │ Weather / Time Data│
              └─────────┬─────────┘
                        │
                 Preprocessing
                        │
             ┌──────────┴──────────┐
             │                     │
             ▼                     ▼
        ┌─────────┐           ┌─────────┐
        │ XGBoost │           │   TCN   │
        └────┬────┘           └────┬────┘
             │                     │
             └──────────┬──────────┘
                        ▼
               ┌─────────────────┐
               │ Hybrid Ensemble │
               └────────┬────────┘
                        ▼
               ┌─────────────────┐
               │ Solar Forecast  │
               └────────┬────────┘
                        ▼
                 FastAPI Backend
                        │
                        ▼
                 React Dashboard

## 💼 Project Highlights

- End-to-end machine learning forecasting pipeline
- Hybrid XGBoost + TCN architecture
- Pre-trained inference-ready models
- FastAPI model serving
- Interactive React dashboard
- Evaluation and error-analysis artifacts
- Modular Python project structure
- Reproducible local deployment

---

## 📌 Project Status

**Prototype / Research Implementation**

The current implementation demonstrates the forecasting architecture
using the available dataset. Deployment on production solar assets
should be preceded by validation using representative real-world
SCADA and weather data.

---

## 👨‍💻 Built With

**Machine Learning**
Python · XGBoost · TensorFlow · TCN · Scikit-learn

**Backend**
FastAPI · Uvicorn

**Frontend**
React · TypeScript · Vite

**Data**
Pandas · NumPy

---

<p align="center">

### ☀️ From environmental data → intelligent forecasting

</p>
