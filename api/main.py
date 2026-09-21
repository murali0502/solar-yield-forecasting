from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import (
    get_predictor,
    router,
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):

    print()
    print("=" * 70)
    print("SOLAR YIELD FORECAST API")
    print("=" * 70)

    print()
    print(
        "Loading ML models..."
    )

    # Load models once during startup.
    get_predictor()

    print()
    print(
        "ML models loaded successfully."
    )

    print(
        "API startup complete."
    )

    print(
        "=" * 70
    )

    yield

    print()
    print(
        "Solar Yield Forecast API shutting down."
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Solar Yield Forecast API",

    description=(
        "Solar yield forecasting API using "
        "XGBoost + TCN hybrid modeling."
    ),

    version="1.0.0",

    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)

# ============================================================
# ROUTES
# ============================================================

app.include_router(
    router,
    prefix="/api",
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "application":
            "Solar Yield Forecast API",

        "version":
            "1.0.0",

        "status":
            "running",

        "model":
            "XGBoost + TCN Hybrid",

        "endpoints": [
            "/api/health",
            "/api/model-info",
            "/api/metrics",
            "/api/predict",
            "/api/predict_day",
            "/api/forecast",
            "/docs",
        ],
    }