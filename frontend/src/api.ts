const API_BASE_URL = "http://127.0.0.1:8000";

export interface HealthResponse {
    status: string;
    models_loaded: boolean;
}

export interface ModelInfoResponse {
    xgboost_weight: number;
    tcn_weight: number;
    tcn_lookback: number;
    feature_count: number;
    features: string[];
}

export interface WeatherObservation {
    timestamp: string;
    temperature: number;
    humidity: number;
    cloud_cover: number;
    wind_speed: number;
    irradiance: number;
}

export interface PredictionResponse {
    timestamp: string;
    xgboost_prediction: number;
    tcn_prediction: number;
    hybrid_prediction: number;
    xgboost_weight: number;
    tcn_weight: number;
}

export interface ForecastPoint {
    timestamp: string;
    xgboost_prediction: number;
    tcn_prediction: number;
    hybrid_prediction: number;
}

export interface ForecastResponse {
    forecast_hours: number;
    xgboost_weight: number;
    tcn_weight: number;
    predictions: ForecastPoint[];
}

async function request<T>(
    endpoint: string,
    options?: RequestInit,
): Promise<T> {
    const response = await fetch(
        `${API_BASE_URL}${endpoint}`,
        {
            ...options,
            headers: {
                "Content-Type": "application/json",
                ...(options?.headers || {}),
            },
        },
    );

    if (!response.ok) {
        const errorText = await response.text();

        throw new Error(
            `API error ${response.status}: ${errorText}`,
        );
    }

    return response.json();
}


export async function getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>(
        "/api/health",
    );
}


export async function getModelInfo(): Promise<ModelInfoResponse> {
    return request<ModelInfoResponse>(
        "/api/model-info",
    );
}


export async function predict(
    observations: WeatherObservation[],
): Promise<PredictionResponse> {
    return request<PredictionResponse>(
        "/api/predict",
        {
            method: "POST",
            body: JSON.stringify({
                observations,
            }),
        },
    );
}


export async function forecast(
    historicalObservations: WeatherObservation[],
    futureObservations: WeatherObservation[],
): Promise<ForecastResponse> {
    return request<ForecastResponse>(
        "/api/forecast",
        {
            method: "POST",
            body: JSON.stringify({
                historical_observations:
                    historicalObservations,

                future_observations:
                    futureObservations,
            }),
        },
    );
}