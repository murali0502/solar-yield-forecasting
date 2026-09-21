import { type FormEvent, useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const FORECAST_YEAR = 2026;

const MONTHS = [
  { value: 1, label: "January" },
  { value: 2, label: "February" },
  { value: 3, label: "March" },
  { value: 4, label: "April" },
  { value: 5, label: "May" },
  { value: 6, label: "June" },
  { value: 7, label: "July" },
  { value: 8, label: "August" },
  { value: 9, label: "September" },
  { value: 10, label: "October" },
  { value: 11, label: "November" },
  { value: 12, label: "December" },
];

type PredictionResponse = {
  hours: number[];
  irradiance: number[];
  xgboost: number[];
  tcn: number[];
  hybrid: number[];
  totals_kwh: {
    xgboost: number;
    tcn: number;
    hybrid: number;
  };
};

type ChartPoint = {
  hour: string;
  xgboost: number;
  tcn: number;
  hybrid: number;
  irradiance: number;
};

type ScenarioResult = {
  cloudCover: number;
  hybrid: number;
  xgboost: number;
  tcn: number;
};

type ModelMetrics = {
  mae: number;
  rmse: number;
  r2: number;
};

type PerformanceMetrics = {
  xgboost: ModelMetrics;
  tcn: ModelMetrics;
  hybrid: ModelMetrics;
};

function App() {
  // ============================================================
  // INPUT STATE
  // ============================================================

  const [month, setMonth] = useState(5);
  const [day, setDay] = useState(30);

  const [temperature, setTemperature] = useState("30");
  const [humidity, setHumidity] = useState(60);
  const [cloudCover, setCloudCover] = useState(20);
  const [windSpeed, setWindSpeed] = useState("3");

  const [dayOfYearInput, setDayOfYearInput] = useState("150");

  // ============================================================
  // THEME STATE
  // ============================================================

  const [theme, setTheme] = useState<"dark" | "light">(() => {
    const savedTheme = localStorage.getItem("solar-theme");

    return savedTheme === "light" ? "light" : "dark";
  });

  // ============================================================
  // APPLICATION STATE
  // ============================================================

  const [prediction, setPrediction] =
    useState<PredictionResponse | null>(null);

  const [loading, setLoading] = useState(false);

  const [backendOnline, setBackendOnline] =
    useState(false);

  const [error, setError] = useState("");

  const [scenarioResults, setScenarioResults] =
    useState<ScenarioResult[]>([]);

  const [scenarioLoading, setScenarioLoading] =
    useState(false);

  const [performanceMetrics, setPerformanceMetrics] =
    useState<PerformanceMetrics | null>(null);

  // ============================================================
  // AI PROCESSING STATE
  //
  // 0 = idle
  // 1 = preparing weather inputs
  // 2 = XGBoost
  // 3 = TCN
  // 4 = Hybrid
  // ============================================================

  const [processingStage, setProcessingStage] =
    useState(0);

  // ============================================================
  // APPLY THEME
  // ============================================================

  useEffect(() => {
    localStorage.setItem(
      "solar-theme",
      theme
    );

    document.documentElement.setAttribute(
      "data-theme",
      theme
    );
  }, [theme]);

  // ============================================================
  // DATE HELPERS
  // ============================================================

  function getDaysInMonth(
    selectedMonth: number
  ): number {
    return new Date(
      FORECAST_YEAR,
      selectedMonth,
      0
    ).getDate();
  }

  function calculateDayOfYear(
    selectedMonth: number,
    selectedDay: number
  ): number {
    const date =
      new Date(
        FORECAST_YEAR,
        selectedMonth - 1,
        selectedDay
      );

    const startOfYear =
      new Date(
        FORECAST_YEAR,
        0,
        1
      );

    return (
      Math.floor(
        (date.getTime() -
          startOfYear.getTime()) /
        (1000 * 60 * 60 * 24)
      ) + 1
    );
  }

  function getDateFromDayOfYear(
    selectedDayOfYear: number
  ): {
    month: number;
    day: number;
  } {
    const safeDayOfYear =
      Math.min(
        365,
        Math.max(
          1,
          Math.round(
            selectedDayOfYear
          )
        )
      );

    const date =
      new Date(
        FORECAST_YEAR,
        0,
        safeDayOfYear
      );

    return {
      month:
        date.getMonth() + 1,
      day:
        date.getDate(),
    };
  }

  function normalizeNumberInput(
    value: string
  ): string {
    if (value === "") {
      return "";
    }

    // Prevent values such as 025 while still
    // allowing normal decimal input such as 25.5.
    return value.replace(
      /^(-?)0+(?=\d)/,
      "$1"
    );
  }

  const daysInSelectedMonth =
    getDaysInMonth(month);

  const dayOfYear =
    calculateDayOfYear(
      month,
      day
    );

  // Keep Day-of-Year synchronized with
  // the selected calendar date.
  useEffect(() => {
    setDayOfYearInput(
      String(dayOfYear)
    );
  }, [dayOfYear]);

  function handleDayOfYearChange(
    value: string
  ) {
    const digits =
      value.replace(
        /\D/g,
        ""
      );

    setDayOfYearInput(
      digits
    );

    if (digits === "") {
      return;
    }

    const numericValue =
      Number(digits);

    if (
      numericValue >= 1 &&
      numericValue <= 365
    ) {
      const nextDate =
        getDateFromDayOfYear(
          numericValue
        );

      setMonth(
        nextDate.month
      );

      setDay(
        nextDate.day
      );
    }
  }

  function handleDayOfYearBlur() {
    const numericValue =
      Number(dayOfYearInput);

    if (
      !Number.isFinite(
        numericValue
      ) ||
      numericValue < 1
    ) {
      const firstDate =
        getDateFromDayOfYear(1);

      setMonth(
        firstDate.month
      );

      setDay(
        firstDate.day
      );

      setDayOfYearInput("1");
      return;
    }

    const safeValue =
      Math.min(
        365,
        Math.round(
          numericValue
        )
      );

    const nextDate =
      getDateFromDayOfYear(
        safeValue
      );

    setMonth(
      nextDate.month
    );

    setDay(
      nextDate.day
    );

    setDayOfYearInput(
      String(safeValue)
    );
  }

  function handleMonthChange(
    nextMonth: number
  ) {
    const maximumDay =
      getDaysInMonth(
        nextMonth
      );

    setMonth(
      nextMonth
    );

    setDay(
      Math.min(
        day,
        maximumDay
      )
    );
  }

  // ============================================================
  // AI PROCESSING ANIMATION
  // ============================================================

  useEffect(() => {
    if (!loading) {
      return;
    }

    setProcessingStage(1);

    const timer =
      window.setInterval(() => {
        setProcessingStage(
          (currentStage) => {
            if (currentStage >= 3) {
              return currentStage;
            }

            return currentStage + 1;
          }
        );
      }, 650);

    return () => {
      window.clearInterval(timer);
    };
  }, [loading]);

  // ============================================================
  // BACKEND CHECK
  // ============================================================

  useEffect(() => {
    checkBackend();
    loadPerformanceMetrics();
  }, []);

  async function checkBackend() {
    try {
      const response =
        await fetch(
          `${API_URL}/api/health`
        );

      if (!response.ok) {
        throw new Error(
          "Backend health check failed"
        );
      }

      const data =
        await response.json();

      setBackendOnline(
        data.status === "healthy" &&
        data.models_loaded === true
      );
    } catch (err) {
      console.error(err);

      setBackendOnline(false);
    }
  }

  async function loadPerformanceMetrics() {
    try {
      const response = await fetch(
        `${API_URL}/api/metrics`
      );

      if (!response.ok) {
        throw new Error("Metrics request failed");
      }

      const data: any = await response.json();

      /*
       * Use the backend's test-set metrics instead of duplicating
       * evaluation numbers in the frontend.
       */
      const readMetric = (
        model: string,
        metric: "mae" | "rmse" | "r2"
      ): number | null => {
        const modelData =
          data?.test?.[model] ??
          data?.models?.[model]?.test ??
          data?.[model]?.test ??
          data?.[model] ??
          null;

        const value =
          modelData?.[metric] ??
          modelData?.[metric.toUpperCase()] ??
          modelData?.metrics?.[metric] ??
          modelData?.metrics?.[metric.toUpperCase()];

        const numericValue = Number(value);

        return Number.isFinite(numericValue)
          ? numericValue
          : null;
      };

      const xgb = {
        mae: readMetric("xgboost", "mae"),
        rmse: readMetric("xgboost", "rmse"),
        r2: readMetric("xgboost", "r2"),
      };

      const tcn = {
        mae: readMetric("tcn", "mae"),
        rmse: readMetric("tcn", "rmse"),
        r2: readMetric("tcn", "r2"),
      };

      const hybrid = {
        mae: readMetric("hybrid", "mae"),
        rmse: readMetric("hybrid", "rmse"),
        r2: readMetric("hybrid", "r2"),
      };

      if (
        xgb.mae !== null &&
        xgb.rmse !== null &&
        xgb.r2 !== null &&
        tcn.mae !== null &&
        tcn.rmse !== null &&
        tcn.r2 !== null &&
        hybrid.mae !== null &&
        hybrid.rmse !== null &&
        hybrid.r2 !== null
      ) {
        setPerformanceMetrics({
          xgboost: {
            mae: xgb.mae,
            rmse: xgb.rmse,
            r2: xgb.r2,
          },
          tcn: {
            mae: tcn.mae,
            rmse: tcn.rmse,
            r2: tcn.r2,
          },
          hybrid: {
            mae: hybrid.mae,
            rmse: hybrid.rmse,
            r2: hybrid.r2,
          },
        });
      }
    } catch (metricsError) {
      console.error(
        "Performance metrics could not be loaded:",
        metricsError
      );
    }
  }

  // ============================================================
  // PREDICTION
  // ============================================================

  async function handlePredict(
    event: FormEvent
  ) {
    event.preventDefault();

    setLoading(true);
    setError("");
    setPrediction(null);
    setProcessingStage(1);

    try {
      if (
        temperature.trim() === "" ||
        windSpeed.trim() === ""
      ) {
        throw new Error(
          "Please enter temperature and wind speed."
        );
      }

      const temperatureValue =
        Number(temperature);

      const windSpeedValue =
        Number(windSpeed);

      if (
        !Number.isFinite(
          temperatureValue
        ) ||
        !Number.isFinite(
          windSpeedValue
        )
      ) {
        throw new Error(
          "Please enter valid numeric weather values."
        );
      }

      if (
        temperatureValue < -50 ||
        temperatureValue > 60
      ) {
        throw new Error(
          "Temperature must be between -50 °C and 60 °C."
        );
      }

      if (
        windSpeedValue < 0 ||
        windSpeedValue > 50
      ) {
        throw new Error(
          "Wind speed must be between 0 and 50 m/s."
        );
      }

      const payload = {
        day_of_year: dayOfYear,
        temperature:
          temperatureValue,
        humidity: Number(
          humidity
        ),
        cloud_cover: Number(
          cloudCover
        ),
        wind_speed:
          windSpeedValue,
      };

      console.log(
        "Forecast request:",
        payload
      );

      const response =
        await fetch(
          `${API_URL}/api/predict_day`,
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",
            },

            body: JSON.stringify(
              payload
            ),
          }
        );

      if (!response.ok) {
        const message =
          await response.text();

        throw new Error(message);
      }

      const data: PredictionResponse =
        await response.json();

      /*
       * Keep the processing animation visible
       * long enough for a client demonstration.
       *
       * This does NOT modify the backend/model.
       */
      await new Promise<void>(
        (resolve) =>
          window.setTimeout(
            resolve,
            1800
          )
      );

      /*
       * All model processing is now complete.
       */
      setProcessingStage(4);

      /*
       * Small pause so the user can actually
       * see the Hybrid completion state.
       */
      await new Promise<void>(
        (resolve) =>
          window.setTimeout(
            resolve,
            350
          )
      );

      console.log(
        "Forecast response:",
        data
      );

      setPrediction(data);

      /*
       * Run the four cloud-cover scenarios using the
       * same date and weather inputs. This is an
       * additional analysis feature and does not
       * retrain or modify any model.
       */
      void runCloudScenarios(payload);

      setBackendOnline(true);
    } catch (err) {
      console.error(err);

      setProcessingStage(0);

      setError(
        err instanceof Error &&
          err.message &&
          !err.message.includes(
            "Failed to fetch"
          )
          ? err.message
          : "Prediction failed. Make sure the FastAPI backend is running on port 8000."
      );
    } finally {
      setLoading(false);
    }
  }

  // ============================================================
  // CLOUD SCENARIO ANALYSIS
  // ============================================================

  async function runCloudScenarios(
    basePayload: {
      day_of_year: number;
      temperature: number;
      humidity: number;
      cloud_cover: number;
      wind_speed: number;
    }
  ) {
    const scenarios = [0, 30, 60, 100];

    setScenarioLoading(true);
    setScenarioResults([]);

    try {
      const results = await Promise.all(
        scenarios.map(async (cloudCoverValue) => {
          const response = await fetch(
            `${API_URL}/api/predict_day`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                ...basePayload,
                cloud_cover: cloudCoverValue,
              }),
            }
          );

          if (!response.ok) {
            throw new Error(
              `Cloud scenario ${cloudCoverValue}% failed`
            );
          }

          const data: PredictionResponse =
            await response.json();

          return {
            cloudCover: cloudCoverValue,
            hybrid:
              Number(data.totals_kwh.hybrid) || 0,
            xgboost:
              Number(data.totals_kwh.xgboost) || 0,
            tcn:
              Number(data.totals_kwh.tcn) || 0,
          };
        })
      );

      setScenarioResults(results);
    } catch (scenarioError) {
      console.error(
        "Cloud scenario analysis failed:",
        scenarioError
      );
      setScenarioResults([]);
    } finally {
      setScenarioLoading(false);
    }
  }

  // ============================================================
  // RESET
  // ============================================================

  function handleReset() {
    setMonth(5);
    setDay(30);

    setTemperature("30");
    setHumidity(60);
    setCloudCover(20);
    setWindSpeed("3");
    setDayOfYearInput("150");

    setPrediction(null);
    setScenarioResults([]);
    setScenarioLoading(false);
    setError("");
    setProcessingStage(0);

    checkBackend();
  }

  // ============================================================
  // CHART DATA
  // ============================================================

  const chartData: ChartPoint[] =
    prediction
      ? prediction.hours.map(
        (hour, index) => ({
          hour: `${String(
            hour
          ).padStart(
            2,
            "0"
          )}:00`,

          xgboost:
            Number(
              prediction.xgboost[
              index
              ]
            ) || 0,

          tcn:
            Number(
              prediction.tcn[
              index
              ]
            ) || 0,

          hybrid:
            Number(
              prediction.hybrid[
              index
              ]
            ) || 0,

          irradiance:
            Number(
              prediction.irradiance[
              index
              ]
            ) || 0,
        })
      )
      : [];

  // ============================================================
  // PEAK FORECAST
  // ============================================================

  let peakForecast = 0;
  let peakIndex = -1;
  let peakHour = "—";

  if (
    prediction &&
    prediction.hybrid.length > 0
  ) {
    peakForecast =
      Math.max(
        ...prediction.hybrid
      );

    peakIndex =
      prediction.hybrid.indexOf(
        peakForecast
      );

    if (peakIndex >= 0) {
      peakHour =
        `${String(
          prediction.hours[
          peakIndex
          ]
        ).padStart(
          2,
          "0"
        )}:00`;
    }
  }

  const forecastHorizon =
    prediction?.hours.length ??
    24;

  const hybridDailyTotal =
    prediction?.totals_kwh.hybrid ?? 0;

  const scenarioBaseline =
    scenarioResults.find(
      (item) => item.cloudCover === cloudCover
    )?.hybrid ?? hybridDailyTotal;

  const scenarioMax =
    scenarioResults.length > 0
      ? Math.max(
        ...scenarioResults.map(
          (item) => item.hybrid
        )
      )
      : 0;

  const scenarioMin =
    scenarioResults.length > 0
      ? Math.min(
        ...scenarioResults.map(
          (item) => item.hybrid
        )
      )
      : 0;

  // ============================================================
  // DATE DISPLAY
  // ============================================================

  const selectedMonthName =
    MONTHS.find(
      (item) =>
        item.value === month
    )?.label ?? "";

  // ============================================================
  // THEME COLORS
  // ============================================================

  const chartTextColor =
    theme === "dark"
      ? "#7180a5"
      : "#667085";

  const chartGridColor =
    theme === "dark"
      ? "rgba(255,255,255,0.055)"
      : "rgba(16,29,52,0.075)";

  const chartAxisColor =
    theme === "dark"
      ? "rgba(255,255,255,0.08)"
      : "rgba(16,29,52,0.10)";

  const tooltipBackground =
    theme === "dark"
      ? "#111a2d"
      : "#ffffff";

  const tooltipBorder =
    theme === "dark"
      ? "rgba(255,255,255,0.12)"
      : "rgba(16,29,52,0.12)";

  const tooltipText =
    theme === "dark"
      ? "#ffffff"
      : "#101827";

  // ============================================================
  // MODEL PROCESSING HELPERS
  // ============================================================

  function getModelState(
    model:
      | "xgboost"
      | "tcn"
      | "hybrid"
  ) {
    if (!loading) {
      return prediction
        ? "complete"
        : "idle";
    }

    if (model === "xgboost") {
      if (processingStage >= 4) {
        return "complete";
      }

      if (processingStage >= 2) {
        return "processing";
      }

      return "waiting";
    }

    if (model === "tcn") {
      if (processingStage >= 4) {
        return "complete";
      }

      if (processingStage >= 3) {
        return "processing";
      }

      return "waiting";
    }

    if (processingStage >= 4) {
      return "complete";
    }

    return "waiting";
  }

  // ============================================================
  // RENDER
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          FRONTEND-ONLY ANIMATION STYLES
          ====================================================== */}

      <style>{`

        /* ======================================================
           SOLAR ENERGY SCENE
           ====================================================== */

        .solar-energy-scene {
          position: relative;
          width: 100%;
          min-height: 350px;
          height: 100%;
          overflow: hidden;
          isolation: isolate;
        }

        .solar-energy-scene .solar-sky-glow {
          position: absolute;
          width: 235px;
          height: 235px;
          top: 0;
          left: 13%;
          border-radius: 50%;
          background:
            radial-gradient(
              circle,
              rgba(242,169,59,0.18) 0%,
              rgba(242,169,59,0.06) 42%,
              transparent 72%
            );
          filter: blur(2px);
          animation:
            skyGlowPulse
            5s
            ease-in-out
            infinite;
          pointer-events: none;
        }

        .solar-energy-scene .solar-source {
          position: absolute;
          top: 6%;
          left: 18%;
          width: 118px;
          height: 118px;
          z-index: 5;
          animation:
            sunBreath
            4.5s
            ease-in-out
            infinite;
        }

        .solar-energy-scene .sun-corona {
          position: absolute;
          inset: -14px;
          border-radius: 50%;
          pointer-events: none;
        }

        .solar-energy-scene .sun-corona-outer {
          background:
            radial-gradient(
              circle,
              rgba(242,169,59,0.16),
              transparent 68%
            );
          animation:
            coronaPulse
            4s
            ease-in-out
            infinite;
        }

        .solar-energy-scene .sun-corona-middle {
          inset: -7px;
          background:
            radial-gradient(
              circle,
              rgba(255,210,116,0.18),
              transparent 68%
            );
        }

        .solar-energy-scene .sun-corona-inner {
          inset: -2px;
          background:
            radial-gradient(
              circle,
              rgba(255,240,180,0.14),
              transparent 72%
            );
        }

        .solar-energy-scene .solar-disc {
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
          border-radius: 50%;
          background:
            radial-gradient(
              circle at 35% 28%,
              #fff2bd 0%,
              #f8c968 14%,
              #e8a632 45%,
              #cc8a20 100%
            );
          box-shadow:
            0 12px 30px rgba(202,133,28,0.26),
            0 0 45px rgba(242,169,59,0.24);
        }

        .solar-energy-scene .solar-disc-highlight {
          position: absolute;
          width: 32%;
          height: 32%;
          top: 12%;
          left: 16%;
          border-radius: 50%;
          background:
            rgba(255,255,255,0.52);
          filter: blur(7px);
        }

        .solar-energy-scene .solar-disc-core {
          position: absolute;
          inset: 7%;
          border-radius: 50%;
          border:
            1px solid rgba(255,230,160,0.22);
        }

        .solar-energy-scene .sunlight-ray {
          position: absolute;
          left: 31%;
          height: 2px;
          transform-origin: left center;
          border-radius: 99px;
          background:
            linear-gradient(
              90deg,
              rgba(242,169,59,0.76),
              rgba(242,169,59,0.22),
              transparent
            );
          filter:
            drop-shadow(
              0 0 5px
              rgba(242,169,59,0.24)
            );
          z-index: 2;
          pointer-events: none;
          animation:
            sunlightPulse
            3s
            ease-in-out
            infinite;
        }

        .solar-energy-scene .ray-one {
          top: 30%;
          width: 49%;
          transform: rotate(17deg);
        }

        .solar-energy-scene .ray-two {
          top: 43%;
          width: 56%;
          transform: rotate(8deg);
          animation-delay: 0.7s;
        }

        .solar-energy-scene .ray-three {
          top: 56%;
          width: 49%;
          transform: rotate(-3deg);
          animation-delay: 1.4s;
        }

        .solar-energy-scene .sunlight-particle {
          position: absolute;
          left: 40%;
          z-index: 4;
          color: #f2a93b;
          font-size: 12px;
          text-shadow:
            0 0 10px
            rgba(242,169,59,0.48);
          animation:
            photonFlow
            3.4s
            linear
            infinite;
          pointer-events: none;
        }

        .solar-energy-scene .particle-one {
          top: 31%;
          animation-delay: 0.1s;
        }

        .solar-energy-scene .particle-two {
          top: 43%;
          animation-delay: 1.2s;
        }

        .solar-energy-scene .particle-three {
          top: 55%;
          animation-delay: 2.25s;
        }

        .solar-energy-scene .solar-panel-visual {
          position: absolute;
          right: 3%;
          bottom: 18%;
          width: 56%;
          height: 180px;
          z-index: 6;
          transform:
            perspective(500px)
            rotateX(7deg)
            rotateY(-7deg)
            rotateZ(-4deg);
          animation:
            panelEnergy
            4.5s
            ease-in-out
            infinite;
        }

        .solar-energy-scene .panel-frame {
          position: absolute;
          inset: 0 0 27px 0;
          padding: 8px;
          border-radius: 8px;
          border:
            2px solid
            rgba(180,193,214,0.36);
          background:
            linear-gradient(
              145deg,
              rgba(23,39,65,0.96),
              rgba(7,15,29,0.98)
            );
          box-shadow:
            0 18px 35px
            rgba(0,0,0,0.22),
            0 0 20px
            rgba(242,169,59,0.08);
          overflow: hidden;
        }

        .solar-energy-scene .panel-grid {
          display: grid;
          grid-template-columns:
            repeat(6, 1fr);
          grid-template-rows:
            repeat(4, 1fr);
          gap: 2px;
          width: 100%;
          height: 100%;
          border-radius: 3px;
          overflow: hidden;
        }

        .solar-energy-scene .panel-cell {
          position: relative;
          display: block;
          background:
            linear-gradient(
              135deg,
              rgba(80,111,154,0.56),
              rgba(25,48,82,0.86)
            );
          border:
            1px solid
            rgba(144,164,193,0.18);
        }

        .solar-energy-scene .panel-cell::after {
          content: "";
          position: absolute;
          inset: 0;
          background:
            linear-gradient(
              115deg,
              rgba(255,255,255,0.10),
              transparent 35%
            );
        }

        .solar-energy-scene .panel-reflection {
          position: absolute;
          width: 32%;
          height: 170%;
          top: -30%;
          left: -40%;
          transform: rotate(20deg);
          background:
            linear-gradient(
              90deg,
              transparent,
              rgba(255,255,255,0.13),
              transparent
            );
          animation:
            panelReflection
            5s
            ease-in-out
            infinite;
        }

        .solar-energy-scene .panel-stand {
          position: absolute;
          width: 28%;
          height: 28px;
          left: 36%;
          bottom: 2px;
          border-left:
            4px solid
            rgba(170,181,201,0.30);
          border-right:
            4px solid
            rgba(170,181,201,0.20);
          transform:
            perspective(200px)
            rotateX(-10deg);
        }

        .solar-energy-scene .energy-flow {
          position: absolute;
          right: 7%;
          bottom: 0;

          display: flex;
          align-items: center;
          justify-content: center;
          gap: 5px;

          padding: 6px 10px;
          min-width: 64px;

          border-radius: 999px;
          border:
            1px solid
            rgba(242,169,59,0.25);

          background:
            rgba(8,15,28,0.92);

          color: #d7a44d;

          font-size: 8px;
          font-weight: 800;
          letter-spacing: 0.08em;

          box-shadow:
            0 7px 18px
            rgba(0,0,0,0.20);

          /* Badge stays perfectly horizontal. */
          transform: translateY(0);

          animation:
            energyPulse
            2.2s
            ease-in-out
            infinite;

          z-index: 8;

        }

        .solar-energy-scene .energy-flow span:first-child {
          font-size: 12px;
        }

        .solar-energy-scene .solar-animation-caption {
          position: absolute;
          left: 12%;
          right: 5%;
          bottom: 3%;
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 9px;
          color: #73819f;
          font-size: 8px;
          font-weight: 800;
          letter-spacing: 0.09em;
          white-space: nowrap;
        }

        .solar-energy-scene
        .solar-animation-caption span:nth-child(odd) {
          color: #a3afc8;
        }

        /* ======================================================
           PROCESSING PANEL
           ====================================================== */

        .forecast-processing {
          margin-bottom: 18px;
          padding: 18px 20px;
          border-radius: 14px;
          border: 1px solid
            rgba(242,169,59,0.18);
          background:
            linear-gradient(
              105deg,
              rgba(242,169,59,0.075),
              rgba(242,169,59,0.025)
            );
          animation:
            processingReveal
            0.35s
            ease-out;
        }

        .forecast-processing-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 15px;
          margin-bottom: 15px;
        }

        .forecast-processing-title {
          display: flex;
          align-items: center;
          gap: 10px;
        }

        .processing-pulse {
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background: #f2a93b;
          box-shadow:
            0 0 0 0
            rgba(242,169,59,0.5);
          animation:
            processingPulse
            1.5s
            infinite;
        }

        .forecast-processing-title strong {
          font-size: 12px;
          letter-spacing: 0.08em;
          text-transform: uppercase;
        }

        .processing-percent {
          font-size: 11px;
          color: #8c99b7;
          font-weight: 700;
        }

        .processing-steps {
          display: grid;
          grid-template-columns:
            repeat(4, minmax(0, 1fr));
          gap: 8px;
        }

        .processing-step {
          position: relative;
          min-height: 62px;
          padding: 10px;
          border-radius: 10px;
          border:
            1px solid
            rgba(255,255,255,0.055);
          background:
            rgba(8,14,27,0.35);
          opacity: 0.48;
          transition:
            opacity 0.25s ease,
            border-color 0.25s ease,
            background 0.25s ease,
            transform 0.25s ease;
        }

        .processing-step.active {
          opacity: 1;
          border-color:
            rgba(242,169,59,0.28);
          background:
            rgba(242,169,59,0.065);
          transform:
            translateY(-1px);
        }

        .processing-step.complete {
          opacity: 1;
          border-color:
            rgba(43,216,121,0.20);
          background:
            rgba(43,216,121,0.045);
        }

        .processing-step-top {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 6px;
          margin-bottom: 8px;
        }

        .processing-step-number {
          font-size: 9px;
          font-weight: 800;
          color: #687594;
        }

        .processing-step-status {
          width: 18px;
          height: 18px;
          display: flex;
          align-items: center;
          justify-content: center;
          border-radius: 50%;
          font-size: 10px;
          font-weight: 800;
        }

        .processing-step-status.waiting {
          border:
            1px solid
            rgba(255,255,255,0.12);
          color: #687594;
        }

        .processing-step-status.processing {
          border:
            1px solid
            rgba(242,169,59,0.35);
          color: #f2a93b;
          animation:
            processingStatusPulse
            1s
            infinite;
        }

        .processing-step-status.complete {
          border:
            1px solid
            rgba(43,216,121,0.30);
          color: #2bd879;
          background:
            rgba(43,216,121,0.08);
        }

        .processing-step-name {
          font-size: 10px;
          color: #9ba8c5;
          font-weight: 700;
          line-height: 1.35;
        }

        .processing-step.active
        .processing-step-name {
          color: #f5f7fc;
        }

        .processing-progress {
          height: 3px;
          margin-top: 14px;
          overflow: hidden;
          border-radius: 99px;
          background:
            rgba(255,255,255,0.06);
        }

        .processing-progress-bar {
          height: 100%;
          border-radius: inherit;
          background:
            linear-gradient(
              90deg,
              #f2a93b,
              #ffd27b
            );
          transition:
            width 0.45s ease;
        }

        /* ======================================================
           MODEL THINKING ANIMATION
           ====================================================== */

        .model-value.model-processing {
          position: relative;
        }

        .model-value.model-processing
        .model-thinking {
          display: inline-flex;
          align-items: center;
          gap: 3px;
          margin-top: 4px;
          color: #8c99b7;
          font-size: 9px;
        }

        .model-thinking-dot {
          width: 4px;
          height: 4px;
          border-radius: 50%;
          background: currentColor;
          animation:
            modelThinking
            1.1s
            infinite;
        }

        .model-thinking-dot:nth-child(2) {
          animation-delay: 0.16s;
        }

        .model-thinking-dot:nth-child(3) {
          animation-delay: 0.32s;
        }

        .model-complete-mark {
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 17px;
          height: 17px;
          margin-top: 4px;
          border-radius: 50%;
          font-size: 10px;
          font-weight: 900;
          color: #2bd879;
          background:
            rgba(43,216,121,0.08);
          border:
            1px solid
            rgba(43,216,121,0.22);
          animation:
            modelComplete
            0.35s
            ease-out;
        }

        .model-value.is-active {
          animation:
            modelCardActive
            0.6s
            ease-out;
        }

        /* ======================================================
           PROCESSING BUTTON
           ====================================================== */

        .predict-button.processing-active {
          position: relative;
          overflow: hidden;
        }

        .predict-button.processing-active::after {
          content: "";
          position: absolute;
          top: 0;
          bottom: 0;
          left: -35%;
          width: 25%;
          background:
            linear-gradient(
              90deg,
              transparent,
              rgba(255,255,255,0.28),
              transparent
            );
          animation:
            buttonSweep
            1.3s
            infinite;
        }

        /* ======================================================
           KEYFRAMES
           ====================================================== */

        @keyframes skyGlowPulse {
          0%,
          100% {
            opacity: 0.65;
            transform: scale(0.96);
          }

          50% {
            opacity: 1;
            transform: scale(1.04);
          }
        }

        @keyframes sunBreath {
          0%,
          100% {
            transform:
              translateY(0)
              scale(1);
          }

          50% {
            transform:
              translateY(-4px)
              scale(1.025);
          }
        }

        @keyframes coronaPulse {
          0%,
          100% {
            opacity: 0.55;
            transform: scale(0.98);
          }

          50% {
            opacity: 1;
            transform: scale(1.04);
          }
        }

        @keyframes sunlightPulse {
          0%,
          100% {
            opacity: 0.25;
          }

          50% {
            opacity: 0.78;
          }
        }

        @keyframes photonFlow {
          0% {
            opacity: 0;
            transform:
              translateX(0)
              translateY(0)
              scale(0.7);
          }

          18% {
            opacity: 1;
          }

          78% {
            opacity: 0.95;
          }

          100% {
            opacity: 0;
            transform:
              translateX(190px)
              translateY(18px)
              scale(1);
          }
        }

        @keyframes panelEnergy {
          0%,
          100% {
            transform:
              perspective(500px)
              rotateX(7deg)
              rotateY(-7deg)
              rotateZ(-4deg)
              translateY(0);
          }

          50% {
            transform:
              perspective(500px)
              rotateX(7deg)
              rotateY(-7deg)
              rotateZ(-4deg)
              translateY(-3px);
          }
        }

        @keyframes panelReflection {
          0%,
          45% {
            left: -45%;
            opacity: 0;
          }

          60% {
            opacity: 0.8;
          }

          90%,
          100% {
            left: 125%;
            opacity: 0;
          }
        }

        @keyframes energyPulse {
          0%,
          100% {
            opacity: 0.68;
            transform:
              translateY(0)
              scale(1);
          }

          50% {
            opacity: 1;
            transform:
              translateY(-1px)
              scale(1.03);
          }
        }

        @keyframes processingReveal {
          from {
            opacity: 0;
            transform:
              translateY(5px);
          }

          to {
            opacity: 1;
            transform:
              translateY(0);
          }
        }

        @keyframes processingPulse {
          0% {
            box-shadow:
              0 0 0 0
              rgba(242,169,59,0.45);
          }

          70% {
            box-shadow:
              0 0 0 7px
              rgba(242,169,59,0);
          }

          100% {
            box-shadow:
              0 0 0 0
              rgba(242,169,59,0);
          }
        }

        @keyframes processingStatusPulse {
          0%,
          100% {
            transform:
              scale(1);
            opacity: 0.65;
          }

          50% {
            transform:
              scale(1.12);
            opacity: 1;
          }
        }

        @keyframes modelThinking {
          0%,
          100% {
            opacity: 0.25;
            transform:
              translateY(0);
          }

          50% {
            opacity: 1;
            transform:
              translateY(-2px);
          }
        }

        @keyframes modelComplete {
          from {
            opacity: 0;
            transform:
              scale(0.65);
          }

          to {
            opacity: 1;
            transform:
              scale(1);
          }
        }

        @keyframes modelCardActive {
          0% {
            transform:
              translateY(0);
          }

          50% {
            transform:
              translateY(-2px);
          }

          100% {
            transform:
              translateY(0);
          }
        }

        @keyframes buttonSweep {
          from {
            left: -35%;
          }

          to {
            left: 120%;
          }
        }

        /* ======================================================
           DATE CONTROLS
           ====================================================== */

        .date-control-grid {
          display: grid;
          grid-template-columns:
            minmax(0, 1fr)
            92px;
          gap: 8px;
          align-items: end;
        }

        .day-count-control {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .day-count-control > span {
          color: #7f8baa;
          font-size: 9px;
          font-weight: 700;
        }

        .day-count-control input {
          width: 100%;
          min-width: 0;
          height: 42px;
          padding: 0 10px;
          border-radius: 9px;
          border:
            1px solid
            rgba(255,255,255,0.08);
          background:
            rgba(255,255,255,0.025);
          color: #f5f7fc;
          font: inherit;
          font-size: 12px;
          font-weight: 700;
          outline: none;
          box-sizing: border-box;
          transition:
            border-color 0.2s ease,
            background 0.2s ease;
        }

        .day-count-control input:focus {
          border-color:
            rgba(242,169,59,0.42);
          background:
            rgba(242,169,59,0.045);
        }

        html[data-theme="light"]
        .day-count-control input {
          border-color:
            rgba(16,29,52,0.10);
          background:
            rgba(16,29,52,0.025);
          color: #101827;
        }

        /* ======================================================
           ADDITIONAL FORECAST FEATURES
           ====================================================== */

        .additional-feature-card {
          margin-top: 18px;
          padding: 20px;
          border-radius: 16px;
          border: 1px solid
            rgba(255,255,255,0.055);
          background:
            rgba(255,255,255,0.018);
        }

        .feature-heading {
          display: flex;
          align-items: flex-start;
          justify-content: space-between;
          gap: 16px;
          margin-bottom: 16px;
        }

        .feature-heading h4 {
          margin: 5px 0 0;
          font-size: 15px;
        }

        .feature-description {
          margin: 6px 0 0;
          color: #7f8baa;
          font-size: 11px;
          line-height: 1.55;
          max-width: 600px;
        }

        .feature-chip {
          flex: 0 0 auto;
          padding: 6px 9px;
          border-radius: 99px;
          border: 1px solid
            rgba(242,169,59,0.18);
          color: #c7a15d;
          background:
            rgba(242,169,59,0.055);
          font-size: 9px;
          font-weight: 800;
          letter-spacing: 0.08em;
        }

        .secondary-chart {
          height: 285px;
          width: 100%;
        }

        .scenario-grid {
          display: grid;
          grid-template-columns:
            repeat(4, minmax(0, 1fr));
          gap: 10px;
        }

        .scenario-card {
          padding: 14px;
          min-height: 135px;
          border-radius: 13px;
          border: 1px solid
            rgba(255,255,255,0.055);
          background:
            rgba(8,14,27,0.28);
          transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            background 0.2s ease;
        }

        .scenario-card:hover {
          transform: translateY(-2px);
          border-color:
            rgba(242,169,59,0.20);
        }

        .scenario-card.selected {
          border-color:
            rgba(242,169,59,0.30);
          background:
            rgba(242,169,59,0.055);
        }

        .scenario-cloud {
          font-size: 18px;
          line-height: 1;
          margin-bottom: 9px;
        }

        .scenario-percent {
          font-size: 17px;
          font-weight: 800;
        }

        .scenario-card > span {
          display: block;
          margin-top: 3px;
          color: #74819f;
          font-size: 9px;
        }

        .scenario-card > strong {
          display: block;
          margin-top: 13px;
          font-size: 14px;
        }

        .scenario-card > small {
          display: block;
          margin-top: 3px;
          color: #74819f;
          font-size: 9px;
        }

        .scenario-summary {
          display: grid;
          grid-template-columns:
            repeat(3, minmax(0, 1fr));
          gap: 10px;
          margin-top: 12px;
          padding-top: 14px;
          border-top:
            1px solid
            rgba(255,255,255,0.055);
        }

        .scenario-summary div {
          display: flex;
          flex-direction: column;
          gap: 5px;
        }

        .scenario-summary span {
          color: #697696;
          font-size: 8px;
          font-weight: 800;
          letter-spacing: 0.07em;
        }

        .scenario-summary strong {
          font-size: 12px;
        }

        html[data-theme="light"]
        .additional-feature-card {
          border-color:
            rgba(16,29,52,0.08);
          background:
            rgba(16,29,52,0.018);
        }

        html[data-theme="light"]
        .scenario-card {
          border-color:
            rgba(16,29,52,0.08);
          background:
            rgba(16,29,52,0.025);
        }

        html[data-theme="light"]
        .scenario-card.selected {
          border-color:
            rgba(183,119,25,0.28);
          background:
            rgba(242,169,59,0.08);
        }

        @media (max-width: 520px) {
          .date-control-grid {
            grid-template-columns: 1fr;
          }

          .solar-energy-scene {
            min-height: 300px;
          }

          .solar-energy-scene .solar-source {
            left: 8%;
            width: 92px;
            height: 92px;
          }

          .solar-energy-scene .solar-panel-visual {
            width: 63%;
            height: 145px;
            right: 1%;
            bottom: 20%;
          }

          .solar-energy-scene .solar-animation-caption {
            gap: 5px;
            font-size: 7px;
          }
        }

        @media (max-width: 700px) {
          .scenario-grid {
            grid-template-columns:
              repeat(2, minmax(0, 1fr));
          }

          .scenario-summary {
            grid-template-columns:
              1fr;
          }

          .feature-heading {
            flex-direction: column;
          }

          .secondary-chart {
            height: 250px;
          }
        }

        @media (max-width: 700px) {
          .processing-steps {
            grid-template-columns:
              repeat(2, minmax(0, 1fr));
          }
        }


        /* ======================================================
           REALISTIC CSS SOLAR PV PANEL
           ====================================================== */

        .solar-energy-scene .pv-panel-visual {
          position: absolute;
          right: 2%;
          bottom: 18%;
          width: 56%;
          height: 205px;
          z-index: 6;

          /* Keep the solar panel completely level and straight. */
          transform: translateY(0);

          animation:
            pvPanelFloat
            6.5s
            ease-in-out
            infinite;

          will-change: transform;

        }

        .solar-energy-scene .pv-panel {
          position: absolute;
          inset: 0 0 29px 0;
          overflow: visible;
          border-radius: 7px;
          padding: 7px;
          box-sizing: border-box;
          background:
            linear-gradient(
              145deg,
              #aeb9c8 0%,
              #536274 5%,
              #1c2b40 9%,
              #0a1424 100%
            );
          border:
            1px solid
            rgba(211,224,241,0.42);
          box-shadow:
            0 22px 42px
            rgba(0,0,0,0.30),
            0 5px 14px
            rgba(0,0,0,0.18),
            inset 0 1px 0
            rgba(255,255,255,0.24);
        }

        .solar-energy-scene .pv-glass {
          position: relative;
          width: 100%;
          height: 100%;
          overflow: hidden;
          border-radius: 3px;
          box-sizing: border-box;
          border:
            1px solid
            rgba(127,164,202,0.28);
          background:
            linear-gradient(
              145deg,
              #173456 0%,
              #0b1c33 42%,
              #07101e 100%
            );
          box-shadow:
            inset 0 0 0 1px
            rgba(255,255,255,0.045),
            inset 0 0 24px
            rgba(52,119,185,0.12);
        }

        .solar-energy-scene .pv-grid {
          display: grid;
          grid-template-columns:
            repeat(6, minmax(0, 1fr));
          grid-template-rows:
            repeat(4, minmax(0, 1fr));
          width: 100%;
          height: 100%;
          gap: 2px;
          padding: 2px;
          box-sizing: border-box;
          background:
            rgba(147,167,192,0.22);
        }

        .solar-energy-scene .pv-cell {
          position: relative;
          overflow: hidden;
          min-width: 0;
          min-height: 0;
          border:
            1px solid
            rgba(112,151,192,0.24);
          background:
            linear-gradient(
              135deg,
              #183b61 0%,
              #0d2948 38%,
              #08192e 72%,
              #0d2038 100%
            );
          box-shadow:
            inset 0 1px 0
            rgba(255,255,255,0.045);
        }

        .solar-energy-scene .pv-cell::before {
          content: "";
          position: absolute;
          top: 0;
          bottom: 0;
          left: 50%;
          width: 1px;
          background:
            rgba(157,186,218,0.12);
        }

        .solar-energy-scene .pv-cell::after {
          content: "";
          position: absolute;
          inset: 0;
          background:
            linear-gradient(
              125deg,
              rgba(255,255,255,0.15),
              transparent 22%,
              transparent 65%,
              rgba(78,137,195,0.07)
            );
        }

        .solar-energy-scene .pv-reflection {
          position: absolute;
          top: -35%;
          left: -55%;
          width: 17%;
          height: 175%;
          transform: rotate(18deg);
          pointer-events: none;
          background:
            linear-gradient(
              90deg,
              transparent,
              rgba(255,255,255,0.30),
              rgba(255,223,156,0.10),
              transparent
            );
          filter: blur(1px);
          opacity: 0;
          animation:
            pvReflection
            8s
            ease-in-out
            infinite;
          z-index: 3;
        }

        .solar-energy-scene .pv-highlight {
          position: absolute;
          inset: 0;
          pointer-events: none;
          z-index: 2;
          background:
            linear-gradient(
              145deg,
              rgba(255,255,255,0.11),
              transparent 20%,
              transparent 70%,
              rgba(67,131,190,0.08)
            );
        }

        .solar-energy-scene .pv-stand {
          position: absolute;
          left: 32%;
          right: 32%;
          bottom: 0;
          height: 30px;
          z-index: -1;
          border-left:
            4px solid
            rgba(154,169,189,0.36);
          border-right:
            4px solid
            rgba(154,169,189,0.26);
          transform:
            perspective(220px)
            rotateX(-9deg);
          filter:
            drop-shadow(
              0 5px 4px
              rgba(0,0,0,0.18)
            );
        }

        .solar-energy-scene .pv-stand::after {
          content: "";
          position: absolute;
          left: -12px;
          right: -12px;
          bottom: -1px;
          height: 4px;
          border-radius: 99px;
          background:
            rgba(128,143,163,0.24);
        }

        .solar-energy-scene .energy-flow {
          position: absolute;
          right: 0;
          bottom: 0;

          display: flex;
          align-items: center;
          justify-content: center;
          gap: 5px;

          padding: 6px 10px;
          min-width: 64px;

          border-radius: 999px;
          border:
            1px solid
            rgba(242,169,59,0.25);
          background:
            rgba(8,15,28,0.92);

          color: #d7a44d;

          font-size: 8px;
          font-weight: 800;

          letter-spacing: 0.08em;

          box-shadow:
            0 7px 18px
            rgba(0,0,0,0.20);

          /* Keep the badge perfectly horizontal. */
          transform: translateY(0);

          animation:
            energyPulse
            2.2s
            ease-in-out
            infinite;

          z-index: 8;
        }

        .solar-energy-scene .energy-flow span:first-child {
          font-size: 12px;
          line-height: 1;
        }

        .solar-energy-scene .solar-animation-caption {
          position: absolute;
          left: 12%;
          right: 5%;
          bottom: 3%;
          display: flex;
          justify-content: center;
          align-items: center;
          gap: 9px;
          color: #73819f;
          font-size: 8px;
          font-weight: 800;
          letter-spacing: 0.09em;
          white-space: nowrap;
        }

        .solar-energy-scene
        .solar-animation-caption span:nth-child(odd) {
          color: #a3afc8;
        }

        @keyframes pvPanelFloat {
          0%,
          100% {
            transform:
              translateY(0);
          }

          50% {
            transform:
              translateY(-4px);
          }
        }

        @keyframes pvReflection {
          0%,
          42% {
            left: -55%;
            opacity: 0;
          }

          52% {
            opacity: 0.75;
          }

          72%,
          100% {
            left: 125%;
            opacity: 0;
          }
        }

        @media (max-width: 850px) {
          .solar-energy-scene .pv-panel-visual {
            width: 62%;
            height: 175px;
          }
        }

        @media (max-width: 650px) {
          .solar-energy-scene .pv-panel-visual {
            width: 66%;
            height: 145px;
            bottom: 16%;
          }

          /* Keep the actual PV panel level on small screens too. */
          .solar-energy-scene .pv-panel {
            transform: none;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .solar-energy-scene .solar-sky-glow,
          .solar-energy-scene .solar-source,
          .solar-energy-scene .sun-corona,
          .solar-energy-scene .sunlight-ray,
          .solar-energy-scene .sunlight-particle,
          .solar-energy-scene .solar-panel-visual,
          .solar-energy-scene .pv-panel-visual,
          .solar-energy-scene .pv-reflection,
          .solar-energy-scene .energy-flow,
          .processing-pulse,
          .processing-step-status.processing,
          .model-thinking-dot,
          .model-complete-mark,
          .model-value.is-active,
          .predict-button.processing-active::after {
            animation: none !important;
          }
        }

      `}</style>

      <div className="background-glow" />

      {/* ======================================================
          HEADER
          ====================================================== */}

      <header className="header">

        <div className="brand">

          <div className="sun-icon">
            <span>☀</span>
          </div>

          <div>
            <h1>
              Solar Yield Forecast
            </h1>

            <p>
              AI-powered solar energy
              forecasting
            </p>
          </div>

        </div>

        <div className="header-actions">

          {/* THEME BUTTON */}

          <button
            type="button"
            className="theme-toggle"
            onClick={() =>
              setTheme(
                theme === "dark"
                  ? "light"
                  : "dark"
              )
            }
            aria-label={`Switch to ${theme === "dark"
              ? "light"
              : "dark"
              } mode`}
          >

            <span className="theme-icon">
              {theme === "dark"
                ? "☀"
                : "☾"}
            </span>

            <span>
              {theme === "dark"
                ? "Light"
                : "Dark"}
            </span>

          </button>

          {/* BACKEND STATUS */}

          <div
            className={`status ${backendOnline
              ? "online"
              : "offline"
              }`}
          >

            <span className="status-dot" />

            {backendOnline
              ? "System Online"
              : "Backend Offline"}

          </div>

        </div>

      </header>

      <main>

        {/* ====================================================
            HERO
            ==================================================== */}

        <section className="hero">

          <div className="hero-content">

            <div className="eyebrow-row">

              <span className="eyebrow">
                SOLAR FORECASTING
              </span>

              <span className="live-line">
                24-HOUR PREDICTION
              </span>

            </div>

            <h2>
              See tomorrow's
              <br />
              <span>
                solar output.
              </span>
            </h2>

            <p>
              Estimate hourly solar
              energy generation using
              weather conditions,
              temporal patterns and a
              Hybrid XGBoost + TCN
              forecasting model.
            </p>

            <div className="hero-meta">

              <span>
                <b>01</b>
                Weather inputs
              </span>

              <span>
                <b>02</b>
                AI prediction
              </span>

              <span>
                <b>03</b>
                Hourly output
              </span>

            </div>

          </div>

          {/* ==================================================
              SOLAR ENERGY ANIMATION
              ================================================== */}

          <div
            className="solar-orbit solar-energy-scene"
            aria-label="Solar energy flow from sunlight to a solar panel"
          >

            <div className="solar-sky-glow" />

            {/* SUN */}

            <div className="hero-sun-large solar-source">

              <div className="sun-corona sun-corona-outer" />
              <div className="sun-corona sun-corona-middle" />
              <div className="sun-corona sun-corona-inner" />

              <div className="solar-disc">
                <div className="solar-disc-highlight" />
                <div className="solar-disc-core" />
              </div>

            </div>

            {/* SUNLIGHT RAYS */}

            <div className="sunlight-ray ray-one" />
            <div className="sunlight-ray ray-two" />
            <div className="sunlight-ray ray-three" />

            {/* MOVING PHOTONS */}

            <span className="sunlight-particle particle-one">
              ✦
            </span>

            <span className="sunlight-particle particle-two">
              ✦
            </span>

            <span className="sunlight-particle particle-three">
              ✦
            </span>

            {/* ==================================================
                SOLAR PV PANEL
                CSS-built photovoltaic panel so the hero has
                no external image dependency.
                ================================================== */}

            <div className="solar-panel-visual pv-panel-visual">

              <div className="pv-panel">

                <div className="pv-glass">

                  <div className="pv-grid">

                    {Array.from({ length: 24 }).map(
                      (_, index) => (
                        <span
                          key={index}
                          className="pv-cell"
                        />
                      )
                    )}

                  </div>

                  <div
                    className="pv-highlight"
                    aria-hidden="true"
                  />

                  <div
                    className="pv-reflection"
                    aria-hidden="true"
                  />

                </div>

              </div>

              <div
                className="pv-stand"
                aria-hidden="true"
              />

              <div className="energy-flow">
                <span>⚡</span>
                <span>ENERGY</span>
              </div>

            </div>


            <div className="solar-animation-caption">

              <span>
                SUNLIGHT
              </span>

              <span>→</span>

              <span>
                SOLAR PANEL
              </span>

              <span>→</span>

              <span>
                ELECTRICITY
              </span>

            </div>

          </div>

        </section>

        {/* ====================================================
            DASHBOARD
            ==================================================== */}

        <section className="dashboard-grid">

          {/* ==================================================
              INPUT PANEL
              ================================================== */}

          <aside className="panel input-panel">

            <div className="panel-heading">

              <div>

                <span className="section-label">
                  FORECAST INPUT
                </span>

                <h3>
                  Weather Conditions
                </h3>

              </div>

              <span className="step-number">
                01
              </span>

            </div>

            <form
              onSubmit={
                handlePredict
              }
            >

              {/* DATE + DAY OF YEAR */}

              <div className="field">

                <div className="field-header">

                  <label>
                    Forecast Date
                  </label>

                  <span className="day-number">
                    Day {dayOfYear}
                  </span>

                </div>

                <div className="date-control-grid">

                  <div className="date-input-row">

                    <select
                      className="date-select"
                      value={month}
                      onChange={(
                        event
                      ) =>
                        handleMonthChange(
                          Number(
                            event.target
                              .value
                          )
                        )
                      }
                    >

                      {MONTHS.map(
                        (
                          monthOption
                        ) => (
                          <option
                            key={
                              monthOption.value
                            }
                            value={
                              monthOption.value
                            }
                          >
                            {
                              monthOption.label
                            }
                          </option>
                        )
                      )}

                    </select>

                    <select
                      className="date-select"
                      value={day}
                      onChange={(
                        event
                      ) =>
                        setDay(
                          Number(
                            event.target
                              .value
                          )
                        )
                      }
                    >

                      {Array.from(
                        {
                          length:
                            daysInSelectedMonth,
                        },
                        (
                          _,
                          index
                        ) =>
                          index + 1
                      ).map(
                        (
                          dayNumber
                        ) => (
                          <option
                            key={
                              dayNumber
                            }
                            value={
                              dayNumber
                            }
                          >
                            {dayNumber}
                          </option>
                        )
                      )}

                    </select>

                  </div>

                  <div className="day-count-control">

                    <span>
                      Day count
                    </span>

                    <input
                      type="number"
                      min="1"
                      max="365"
                      inputMode="numeric"
                      value={
                        dayOfYearInput
                      }
                      onChange={(
                        event
                      ) =>
                        handleDayOfYearChange(
                          event.target.value
                        )
                      }
                      onBlur={
                        handleDayOfYearBlur
                      }
                      aria-label="Day of year"
                    />

                  </div>

                </div>

                <small className="calculated-day">
                  {selectedMonthName}{" "}
                  {day},{" "}
                  {FORECAST_YEAR}
                  {" · "}
                  Day {dayOfYear} of 365
                </small>

              </div>

              {/* TEMPERATURE */}

              <div className="field">

                <label>
                  Temperature
                </label>

                <div className="input-with-unit">

                  <input
                    type="number"
                    inputMode="decimal"
                    value={
                      temperature
                    }
                    onChange={(
                      event
                    ) =>
                      setTemperature(
                        normalizeNumberInput(
                          event.target
                            .value
                        )
                      )
                    }
                    step="0.1"
                  />

                  <span>
                    °C
                  </span>

                </div>

              </div>

              {/* HUMIDITY */}

              <div className="field">

                <div className="field-header">

                  <label>
                    Humidity
                  </label>

                  <span>
                    {humidity}%
                  </span>

                </div>

                <input
                  className="range-input"
                  type="range"
                  min="0"
                  max="100"
                  value={
                    humidity
                  }
                  onChange={(
                    event
                  ) =>
                    setHumidity(
                      Number(
                        event.target
                          .value
                      )
                    )
                  }
                />

                <div className="range-labels">

                  <span>
                    0%
                  </span>

                  <span>
                    100%
                  </span>

                </div>

              </div>

              {/* CLOUD COVER */}

              <div className="field">

                <div className="field-header">

                  <label>
                    Cloud Cover
                  </label>

                  <span>
                    {cloudCover}%
                  </span>

                </div>

                <input
                  className="range-input"
                  type="range"
                  min="0"
                  max="100"
                  value={
                    cloudCover
                  }
                  onChange={(
                    event
                  ) =>
                    setCloudCover(
                      Number(
                        event.target
                          .value
                      )
                    )
                  }
                />

                <div className="range-labels">

                  <span>
                    Clear
                  </span>

                  <span>
                    Overcast
                  </span>

                </div>

              </div>

              {/* WIND */}

              <div className="field">

                <label>
                  Wind Speed
                </label>

                <div className="input-with-unit">

                  <input
                    type="number"
                    min="0"
                    inputMode="decimal"
                    value={
                      windSpeed
                    }
                    onChange={(
                      event
                    ) =>
                      setWindSpeed(
                        normalizeNumberInput(
                          event.target
                            .value
                        )
                      )
                    }
                    step="0.1"
                  />

                  <span>
                    m/s
                  </span>

                </div>

              </div>

              {/* BUTTONS */}

              <div className="button-row">

                <button
                  className={`predict-button ${loading
                    ? "processing-active"
                    : ""
                    }`}
                  type="submit"
                  disabled={
                    loading
                  }
                >

                  {loading ? (
                    <>
                      <span className="loading-spinner" />

                      Generating
                      forecast
                    </>
                  ) : (
                    <>
                      Generate Forecast

                      <span className="button-arrow">
                        →
                      </span>
                    </>
                  )}

                </button>

                <button
                  className="reset-button"
                  type="button"
                  onClick={
                    handleReset
                  }
                  disabled={
                    loading
                  }
                >
                  Reset inputs
                </button>

              </div>

            </form>

            {error && (
              <div className="error">

                <span>
                  !
                </span>

                <div>

                  <strong>
                    Forecast unavailable
                  </strong>

                  <p>
                    {error}
                  </p>

                </div>

              </div>
            )}

          </aside>

          {/* ==================================================
              RESULTS
              ================================================== */}

          <div className="results-panel">

            {/* RESULTS HEADER */}

            <div className="results-header">

              <div>

                <span className="section-label">
                  DAILY FORECAST
                </span>

                <h3>
                  {prediction
                    ? `${selectedMonthName} ${day}`
                    : "Your solar outlook"}
                </h3>

              </div>

              {prediction && (
                <div className="result-date">
                  {selectedMonthName}{" "}
                  {day},{" "}
                  {FORECAST_YEAR}
                </div>
              )}

            </div>

            {/* ==================================================
                AI PROCESSING SEQUENCE
                ================================================== */}

            {loading && (
              <div className="forecast-processing">

                <div className="forecast-processing-header">

                  <div className="forecast-processing-title">

                    <span className="processing-pulse" />

                    <strong>
                      AI Forecast Processing
                    </strong>

                  </div>

                  <span className="processing-percent">

                    {processingStage >= 4
                      ? "100%"
                      : processingStage === 3
                        ? "75%"
                        : processingStage === 2
                          ? "50%"
                          : "25%"}

                  </span>

                </div>

                <div className="processing-steps">

                  {/* STEP 1 */}

                  <div
                    className={`processing-step ${processingStage >= 1
                      ? processingStage >= 2
                        ? "complete"
                        : "active"
                      : ""
                      }`}
                  >

                    <div className="processing-step-top">

                      <span className="processing-step-number">
                        01
                      </span>

                      <span
                        className={`processing-step-status ${processingStage >= 2
                          ? "complete"
                          : processingStage >= 1
                            ? "processing"
                            : "waiting"
                          }`}
                      >
                        {processingStage >= 2
                          ? "✓"
                          : "•"}
                      </span>

                    </div>

                    <div className="processing-step-name">
                      Weather inputs
                    </div>

                  </div>

                  {/* STEP 2 */}

                  <div
                    className={`processing-step ${processingStage >= 2
                      ? processingStage >= 3
                        ? "complete"
                        : "active"
                      : ""
                      }`}
                  >

                    <div className="processing-step-top">

                      <span className="processing-step-number">
                        02
                      </span>

                      <span
                        className={`processing-step-status ${processingStage >= 3
                          ? "complete"
                          : processingStage >= 2
                            ? "processing"
                            : "waiting"
                          }`}
                      >
                        {processingStage >= 3
                          ? "✓"
                          : "•"}
                      </span>

                    </div>

                    <div className="processing-step-name">
                      XGBoost analysis
                    </div>

                  </div>

                  {/* STEP 3 */}

                  <div
                    className={`processing-step ${processingStage >= 3
                      ? processingStage >= 4
                        ? "complete"
                        : "active"
                      : ""
                      }`}
                  >

                    <div className="processing-step-top">

                      <span className="processing-step-number">
                        03
                      </span>

                      <span
                        className={`processing-step-status ${processingStage >= 4
                          ? "complete"
                          : processingStage >= 3
                            ? "processing"
                            : "waiting"
                          }`}
                      >
                        {processingStage >= 4
                          ? "✓"
                          : "•"}
                      </span>

                    </div>

                    <div className="processing-step-name">
                      TCN temporal analysis
                    </div>

                  </div>

                  {/* STEP 4 */}

                  <div
                    className={`processing-step ${processingStage >= 4
                      ? "complete"
                      : ""
                      }`}
                  >

                    <div className="processing-step-top">

                      <span className="processing-step-number">
                        04
                      </span>

                      <span
                        className={`processing-step-status ${processingStage >= 4
                          ? "complete"
                          : "waiting"
                          }`}
                      >
                        {processingStage >= 4
                          ? "✓"
                          : "•"}
                      </span>

                    </div>

                    <div className="processing-step-name">
                      Hybrid forecast
                    </div>

                  </div>

                </div>

                <div className="processing-progress">

                  <div
                    className="processing-progress-bar"
                    style={{
                      width: `${processingStage >= 4
                        ? 100
                        : processingStage * 25
                        }%`,
                    }}
                  />

                </div>

              </div>
            )}

            {/* ==================================================
                PRIMARY FORECAST
                ================================================== */}

            <div className="primary-result">

              <div className="primary-result-main">

                <span className="primary-label">
                  EXPECTED DAILY OUTPUT
                </span>

                <div className="primary-value">

                  <strong>
                    {prediction
                      ? prediction
                        .totals_kwh
                        .hybrid.toLocaleString(
                          "en-IN",
                          {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2,
                          }
                        )
                      : "—"}
                  </strong>

                  <span>
                    kWh
                  </span>

                </div>

                <p>
                  Final Hybrid forecast
                </p>

              </div>

              <div className="peak-result">

                <div className="peak-icon">
                  ↗
                </div>

                <div>

                  <span>
                    PEAK OUTPUT
                  </span>

                  <strong>
                    {prediction
                      ? `${peakForecast.toFixed(
                        2
                      )} W`
                      : "—"}
                  </strong>

                  <small>
                    {prediction
                      ? `at ${peakHour}`
                      : "awaiting forecast"}
                  </small>

                </div>

              </div>

            </div>

            {/* ==================================================
                MODEL OUTPUTS
                ================================================== */}

            <div className="model-strip">

              <div className="model-strip-title">
                MODEL OUTPUTS
              </div>

              <div className="model-values">

                {/* XGBOOST */}

                <div
                  className={`model-value xgb ${getModelState(
                    "xgboost"
                  ) === "processing"
                    ? "model-processing is-active"
                    : ""
                    }`}
                >

                  <span className="model-dot" />

                  <div>

                    <small>
                      XGBoost
                    </small>

                    <strong>
                      {prediction
                        ? prediction
                          .totals_kwh
                          .xgboost.toFixed(
                            2
                          )
                        : loading
                          ? "..."
                          : "—"}
                    </strong>

                    <em>
                      kWh
                    </em>

                    {loading &&
                      getModelState(
                        "xgboost"
                      ) ===
                      "processing" && (
                        <span className="model-thinking">

                          <span className="model-thinking-dot" />
                          <span className="model-thinking-dot" />
                          <span className="model-thinking-dot" />

                        </span>
                      )}

                    {loading &&
                      getModelState(
                        "xgboost"
                      ) ===
                      "complete" && (
                        <span className="model-complete-mark">
                          ✓
                        </span>
                      )}

                  </div>

                </div>

                {/* TCN */}

                <div
                  className={`model-value tcn ${getModelState(
                    "tcn"
                  ) === "processing"
                    ? "model-processing is-active"
                    : ""
                    }`}
                >

                  <span className="model-dot" />

                  <div>

                    <small>
                      TCN
                    </small>

                    <strong>
                      {prediction
                        ? prediction
                          .totals_kwh
                          .tcn.toFixed(
                            2
                          )
                        : loading
                          ? "..."
                          : "—"}
                    </strong>

                    <em>
                      kWh
                    </em>

                    {loading &&
                      getModelState(
                        "tcn"
                      ) ===
                      "processing" && (
                        <span className="model-thinking">

                          <span className="model-thinking-dot" />
                          <span className="model-thinking-dot" />
                          <span className="model-thinking-dot" />

                        </span>
                      )}

                    {loading &&
                      getModelState(
                        "tcn"
                      ) ===
                      "complete" && (
                        <span className="model-complete-mark">
                          ✓
                        </span>
                      )}

                  </div>

                </div>

                {/* HYBRID */}

                <div
                  className={`model-value hybrid ${getModelState(
                    "hybrid"
                  ) === "complete"
                    ? "is-active"
                    : ""
                    }`}
                >

                  <span className="model-dot" />

                  <div>

                    <small>
                      HYBRID
                    </small>

                    <strong>
                      {prediction
                        ? prediction
                          .totals_kwh
                          .hybrid.toFixed(
                            2
                          )
                        : loading
                          ? "..."
                          : "—"}
                    </strong>

                    <em>
                      kWh
                    </em>

                    {loading &&
                      getModelState(
                        "hybrid"
                      ) ===
                      "waiting" && (
                        <span className="model-thinking">

                          <span className="model-thinking-dot" />
                          <span className="model-thinking-dot" />
                          <span className="model-thinking-dot" />

                        </span>
                      )}

                    {loading &&
                      getModelState(
                        "hybrid"
                      ) ===
                      "complete" && (
                        <span className="model-complete-mark">
                          ✓
                        </span>
                      )}

                  </div>

                </div>

              </div>

            </div>

            {/* ==================================================
                INSIGHT
                ================================================== */}

            {prediction && (
              <div className="forecast-insight">

                <div className="insight-icon">
                  ☀
                </div>

                <div>

                  <span>
                    FORECAST INSIGHT
                  </span>

                  <p>
                    Peak solar production
                    is expected around{" "}
                    <strong>
                      {peakHour}
                    </strong>
                    , reaching
                    approximately{" "}
                    <strong>
                      {peakForecast.toFixed(
                        2
                      )} W
                    </strong>
                    .
                  </p>

                </div>

              </div>
            )}

            {/* ==================================================
                CHART
                ================================================== */}

            <div className="chart-card">

              {prediction && (
                <div className="day-night-section">

                  <div className="day-night-labels">

                    <span>
                      OVERNIGHT
                    </span>

                    <span>
                      DAYLIGHT
                    </span>

                    <span>
                      NIGHT
                    </span>

                  </div>

                  <div className="day-night-bar">

                    {prediction.hours.map(
                      (hour) => {

                        const isDay =
                          hour >= 6 &&
                          hour < 18;

                        return (
                          <div
                            key={hour}
                            className={`hour-block ${isDay
                              ? "day"
                              : "night"
                              }`}
                            title={`${String(
                              hour
                            ).padStart(
                              2,
                              "0"
                            )}:00 — ${isDay
                              ? "Daylight"
                              : "Night"
                              }`}
                          >

                            <span>
                              {String(
                                hour
                              ).padStart(
                                2,
                                "0"
                              )}
                            </span>

                          </div>
                        );
                      }
                    )}

                  </div>

                </div>
              )}

              <div className="chart-header">

                <div>

                  <span className="section-label">
                    HOURLY OUTPUT
                  </span>

                  <h4>
                    Solar generation through
                    the day
                  </h4>

                </div>

                {prediction && (
                  <span className="chart-date">
                    {forecastHorizon} hours
                  </span>
                )}

              </div>

              <div className="chart">

                {prediction ? (

                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                  >

                    <AreaChart
                      data={chartData}
                      margin={{
                        top: 15,
                        right: 10,
                        left: 0,
                        bottom: 5,
                      }}
                    >

                      <defs>

                        <linearGradient
                          id="hybridGradient"
                          x1="0"
                          y1="0"
                          x2="0"
                          y2="1"
                        >

                          <stop
                            offset="0%"
                            stopColor="#f2a93b"
                            stopOpacity={
                              theme ===
                                "dark"
                                ? 0.30
                                : 0.20
                            }
                          />

                          <stop
                            offset="65%"
                            stopColor="#f2a93b"
                            stopOpacity={
                              theme ===
                                "dark"
                                ? 0.08
                                : 0.045
                            }
                          />

                          <stop
                            offset="100%"
                            stopColor="#f2a93b"
                            stopOpacity={0}
                          />

                        </linearGradient>

                      </defs>

                      <CartesianGrid
                        strokeDasharray="2 5"
                        stroke={
                          chartGridColor
                        }
                        vertical={false}
                      />

                      <XAxis
                        dataKey="hour"
                        tick={{
                          fill:
                            chartTextColor,
                          fontSize: 10,
                        }}
                        tickLine={false}
                        axisLine={{
                          stroke:
                            chartAxisColor,
                        }}
                        interval={2}
                      />

                      <YAxis
                        tick={{
                          fill:
                            chartTextColor,
                          fontSize: 10,
                        }}
                        tickLine={false}
                        axisLine={false}
                        width={48}
                        label={{
                          value: "W",
                          angle: -90,
                          position:
                            "insideLeft",
                          fill:
                            chartTextColor,
                          fontSize: 10,
                        }}
                      />

                      <Tooltip
                        cursor={{
                          stroke:
                            "rgba(242,169,59,0.30)",
                          strokeWidth: 1,
                        }}
                        contentStyle={{
                          background:
                            tooltipBackground,
                          border:
                            `1px solid ${tooltipBorder}`,
                          borderRadius: 10,
                          color:
                            tooltipText,
                          boxShadow:
                            theme ===
                              "dark"
                              ? "0 15px 40px rgba(0,0,0,0.35)"
                              : "0 15px 40px rgba(16,29,52,0.12)",
                        }}
                        labelStyle={{
                          color:
                            tooltipText,
                          fontWeight: 700,
                          marginBottom: 7,
                        }}
                        formatter={(
                          value,
                          name
                        ) => [
                            `${Number(
                              value ?? 0
                            ).toFixed(
                              2
                            )} W`,
                            String(name),
                          ]}
                      />

                      <Legend />

                      {/* HYBRID */}

                      <Area
                        type="monotone"
                        dataKey="hybrid"
                        name="Hybrid"
                        stroke="#f2a93b"
                        strokeWidth={3}
                        fill="url(#hybridGradient)"
                        fillOpacity={1}
                        dot={false}
                        activeDot={{
                          r: 5,
                          strokeWidth: 2,
                        }}
                        isAnimationActive={true}
                        animationDuration={1200}
                        animationEasing="ease-out"
                      />

                      {/* XGBOOST */}

                      <Line
                        type="monotone"
                        dataKey="xgboost"
                        name="XGBoost"
                        stroke="#35c9b0"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{
                          r: 4,
                        }}
                        isAnimationActive={true}
                        animationDuration={900}
                      />

                      {/* TCN */}

                      <Line
                        type="monotone"
                        dataKey="tcn"
                        name="TCN"
                        stroke="#b28cf0"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{
                          r: 4,
                        }}
                        isAnimationActive={true}
                        animationDuration={1000}
                      />

                    </AreaChart>

                  </ResponsiveContainer>

                ) : (

                  <div className="empty-chart">

                    <div className="empty-sun">
                      ☀
                    </div>

                    <strong>
                      {loading
                        ? "Processing your forecast..."
                        : "Your forecast will appear here"}
                    </strong>

                    <p>
                      {loading
                        ? "The forecasting models are analyzing the selected conditions."
                        : "Enter the weather conditions and generate a 24-hour solar production forecast."}
                    </p>

                  </div>

                )}

              </div>

            </div>

            {/* ==================================================
                ADDITIONAL FEATURE — SOLAR GENERATION CURVE
                ================================================== */}

            {prediction && (
              <div className="additional-feature-card solar-curve-feature">

                <div className="feature-heading">

                  <div>
                    <span className="section-label">
                      SOLAR GENERATION CURVE
                    </span>

                    <h4>
                      Irradiance and expected production
                    </h4>

                    <p className="feature-description">
                      Daylight irradiance is shown alongside
                      the final Hybrid forecast to visualize
                      the solar generation pattern.
                    </p>
                  </div>

                  <span className="feature-chip">
                    24 HOURS
                  </span>

                </div>

                <div className="secondary-chart">

                  <ResponsiveContainer
                    width="100%"
                    height="100%"
                  >

                    <LineChart
                      data={chartData}
                      margin={{
                        top: 15,
                        right: 10,
                        left: 0,
                        bottom: 5,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="2 5"
                        stroke={chartGridColor}
                        vertical={false}
                      />

                      <XAxis
                        dataKey="hour"
                        tick={{
                          fill: chartTextColor,
                          fontSize: 10,
                        }}
                        tickLine={false}
                        axisLine={{
                          stroke: chartAxisColor,
                        }}
                        interval={2}
                      />

                      <YAxis
                        yAxisId="irradiance"
                        orientation="left"
                        tick={{
                          fill: chartTextColor,
                          fontSize: 10,
                        }}
                        tickLine={false}
                        axisLine={false}
                        width={58}
                        label={{
                          value: "W/m²",
                          angle: -90,
                          position: "insideLeft",
                          fill: chartTextColor,
                          fontSize: 9,
                        }}
                      />

                      <YAxis
                        yAxisId="generation"
                        orientation="right"
                        tick={{
                          fill: chartTextColor,
                          fontSize: 10,
                        }}
                        tickLine={false}
                        axisLine={false}
                        width={48}
                        label={{
                          value: "W",
                          angle: 90,
                          position: "insideRight",
                          fill: chartTextColor,
                          fontSize: 9,
                        }}
                      />

                      <Tooltip
                        contentStyle={{
                          background: tooltipBackground,
                          border:
                            `1px solid ${tooltipBorder}`,
                          borderRadius: 10,
                          color: tooltipText,
                        }}
                        labelStyle={{
                          color: tooltipText,
                          fontWeight: 700,
                          marginBottom: 7,
                        }}

                        formatter={(value, name) => [
                          `${Number(
                            value ?? 0
                          ).toFixed(2)} ${String(name) == "Irradiance"
                            ? "W/m²"
                            : "W"
                          }`,
                          String(name),
                        ]}
                      />

                      <Legend />

                      <Line
                        yAxisId="irradiance"
                        type="monotone"
                        dataKey="irradiance"
                        name="Irradiance"
                        stroke="#f2a93b"
                        strokeWidth={2}
                        dot={false}
                        activeDot={{ r: 4 }}
                        isAnimationActive={true}
                        animationDuration={900}
                      />

                      <Line
                        yAxisId="generation"
                        type="monotone"
                        dataKey="hybrid"
                        name="Hybrid Generation"
                        stroke="#35c9b0"
                        strokeWidth={3}
                        dot={false}
                        activeDot={{ r: 5 }}
                        isAnimationActive={true}
                        animationDuration={1100}
                      />

                    </LineChart>

                  </ResponsiveContainer>

                </div>

              </div>
            )}

            {/* ==================================================
                ADDITIONAL FEATURE — CLOUD SCENARIO ANALYSIS
                ================================================== */}

            <div className="additional-feature-card cloud-scenario-feature">

              <div className="feature-heading">

                <div>
                  <span className="section-label">
                    CLOUD SCENARIO ANALYSIS
                  </span>

                  <h4>
                    What happens as cloud cover changes?
                  </h4>

                  <p className="feature-description">
                    The same forecasting pipeline is evaluated
                    under four cloud-cover scenarios.
                  </p>
                </div>

                <span className="feature-chip">
                  HYBRID
                </span>

              </div>

              <div className="scenario-grid">

                {[0, 30, 60, 100].map(
                  (scenarioCloud) => {
                    const result =
                      scenarioResults.find(
                        (item) =>
                          item.cloudCover ===
                          scenarioCloud
                      );

                    return (
                      <div
                        key={scenarioCloud}
                        className={`scenario-card ${scenarioCloud === cloudCover
                          ? "selected"
                          : ""
                          }`}
                      >

                        <div className="scenario-cloud">
                          {scenarioCloud === 0
                            ? "☀"
                            : scenarioCloud === 100
                              ? "☁"
                              : "⛅"}
                        </div>

                        <div className="scenario-percent">
                          {scenarioCloud}%
                        </div>

                        <span>
                          Cloud cover
                        </span>

                        <strong>
                          {scenarioLoading
                            ? "..."
                            : result
                              ? `${result.hybrid.toFixed(
                                1
                              )} kWh`
                              : "—"}
                        </strong>

                        <small>
                          Hybrid daily output
                        </small>

                      </div>
                    );
                  }
                )}

              </div>

              {scenarioResults.length > 0 && (
                <div className="scenario-summary">

                  <div>
                    <span>
                      CURRENT SCENARIO
                    </span>
                    <strong>
                      {cloudCover}% ·{" "}
                      {scenarioBaseline.toFixed(2)} kWh
                    </strong>
                  </div>

                  <div>
                    <span>
                      CLEAR-SKY POTENTIAL
                    </span>
                    <strong>
                      {scenarioMax.toFixed(2)} kWh
                    </strong>
                  </div>

                  <div>
                    <span>
                      OVERCAST SCENARIO
                    </span>
                    <strong>
                      {scenarioMin.toFixed(2)} kWh
                    </strong>
                  </div>

                </div>
              )}

            </div>

            {/* ==================================================
                PERFORMANCE
                ================================================== */}

            <div className="performance-section">

              <div className="performance-heading">

                <div>

                  <span className="section-label">
                    MODEL PERFORMANCE
                  </span>

                  <h4>
                    Forecasting model
                    accuracy
                  </h4>

                </div>

                <span>
                  Test-set evaluation
                </span>

              </div>

              <div className="performance-grid">

                {/* XGBOOST */}

                <div className="performance-card xgb">

                  <div className="performance-top">

                    <span className="model-indicator" />

                    <strong>
                      XGBoost
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      MAE
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.xgboost.mae.toFixed(2) : "—"}
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      RMSE
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.xgboost.rmse.toFixed(2) : "—"}
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      R²
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.xgboost.r2.toFixed(3) : "—"}
                    </strong>

                  </div>

                </div>

                {/* TCN */}

                <div className="performance-card tcn">

                  <div className="performance-top">

                    <span className="model-indicator" />

                    <strong>
                      TCN
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      MAE
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.tcn.mae.toFixed(2) : "—"}
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      RMSE
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.tcn.rmse.toFixed(2) : "—"}
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      R²
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.tcn.r2.toFixed(3) : "—"}
                    </strong>

                  </div>

                </div>

                {/* HYBRID */}

                <div className="performance-card hybrid">

                  <div className="performance-top">

                    <span className="model-indicator" />

                    <strong>
                      Hybrid
                    </strong>

                    <span className="best-badge">
                      BEST R²
                    </span>

                  </div>

                  <div className="metric-row">

                    <span>
                      MAE
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.hybrid.mae.toFixed(2) : "—"}
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      RMSE
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.hybrid.rmse.toFixed(2) : "—"}
                    </strong>

                  </div>

                  <div className="metric-row">

                    <span>
                      R²
                    </span>

                    <strong>
                      {performanceMetrics ? performanceMetrics.hybrid.r2.toFixed(3) : "—"}
                    </strong>

                  </div>

                </div>

              </div>

            </div>

          </div>

        </section>

      </main>

      {/* ======================================================
          FOOTER
          ====================================================== */}

      <footer>

        <span>
          Solar Yield Forecast
        </span>

        <span>
          XGBoost + TCN Hybrid
          Forecasting
        </span>

      </footer>

    </div>
  );
}

export default App;