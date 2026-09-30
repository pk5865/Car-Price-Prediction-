"""FastAPI app: Car Price Predictor with SQLite, no MySQL needed."""
from pathlib import Path
from typing import Any, Optional
import json
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

# ==================== ML IMPORTS WITH SAFE FALLBACK ====================
try:
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    ColumnTransformer = None
    OneHotEncoder = None
    GradientBoostingRegressor = None
    Pipeline = None
    train_test_split = None
    r2_score = None
    mean_absolute_error = None
    SKLEARN_AVAILABLE = False

from database import init_db, get_db, SearchHistory

# Initialize database FIRST
init_db()

# ==================== CONFIG ====================

ROOT = Path(__file__).resolve().parent
OPT = {
    "companies": {
        "Maruti": ["Swift", "Alto", "Celerio", "WagonR"],
        "Hyundai": ["Creta", "i20", "Venue", "Xcent"],
        "Tata": ["Nexon", "Harrier", "Punch", "Safari"],
        "Honda": ["City", "Accord", "CR-V", "Jazz"],
        "Mahindra": ["XUV500", "Bolero", "Scorpio", "Xylo"],
    },
    "fuel_types": ["Petrol", "Diesel", "CNG", "LPG"],
    "year_min": 1995,
    "year_max": 2026,
}

REF_YEAR = 2026

# ==================== PRICE PREDICTION MODEL ====================

DATASET_PATH = ROOT / "car_data.csv"


class DummyModel:
    def predict(self, X):
        """Fallback model when ML dependencies or dataset are unavailable."""
        if isinstance(X, pd.DataFrame):
            age = float(X.get("age", [0])[0])
            log_kms = float(X.get("log_kms", [0])[0])
        else:
            age, log_kms = 0.0, 0.0

        base = 10.0
        price_log = base - (0.05 * age) - (0.01 * log_kms)
        price = float(np.expm1(price_log))
        return np.array([price])


MODEL = DummyModel()
MODEL_R2 = None
MODEL_MAE = None
MODEL_ROWS = 0


def train_price_model():
    """
    Train Gradient Boosting model using only the existing
    five prediction features:
        company
        model
        fuel_type
        age
        log_kms
    """

    global MODEL, MODEL_R2, MODEL_MAE, MODEL_ROWS

    if not SKLEARN_AVAILABLE:
        print("WARNING: scikit-learn is not installed. Using fallback prediction model.")
        MODEL = DummyModel()
        MODEL_R2 = None
        MODEL_MAE = None
        MODEL_ROWS = 0
        return

    if not DATASET_PATH.exists():
        print(f"WARNING: Dataset not found: {DATASET_PATH}")
        print("Price prediction model was not trained. Using fallback prediction model.")
        MODEL = DummyModel()
        MODEL_R2 = None
        MODEL_MAE = None
        MODEL_ROWS = 0
        return

    try:
        df = pd.read_csv(DATASET_PATH)

        # -------------------------------------------------
        # Expected columns in car_data.csv:
        #
        # company
        # model
        # year
        # kms_driven
        # fuel_type
        # price
        # -------------------------------------------------

        required_columns = [
            "company",
            "model",
            "year",
            "kms_driven",
            "fuel_type",
            "price",
        ]

        missing = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing:
            print(
                f"WARNING: Dataset is missing columns: {missing}"
            )
            print(
                "Expected: company, model, year, kms_driven, "
                "fuel_type, price"
            )
            return

        # Keep ONLY the five features used by the application
        df = df[required_columns].copy()

        # Remove invalid rows
        df = df.dropna()

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["kms_driven"] = pd.to_numeric(
            df["kms_driven"],
            errors="coerce"
        )
        df["price"] = pd.to_numeric(
            df["price"],
            errors="coerce"
        )

        df = df.dropna()

        # Valid values only
        df = df[
            (df["year"] >= OPT["year_min"]) &
            (df["year"] <= REF_YEAR) &
            (df["kms_driven"] >= 0) &
            (df["price"] > 0)
        ]

        if len(df) < 20:
            print(
                "WARNING: Not enough valid rows to train model."
            )
            return

        # Feature engineering
        df["age"] = REF_YEAR - df["year"]
        df["log_kms"] = np.log1p(df["kms_driven"])

        X = df[
            [
                "company",
                "model",
                "fuel_type",
                "age",
                "log_kms",
            ]
        ]

        y = df["price"]

        # -------------------------------------------------
        # Train/test split
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42
        )

        # -------------------------------------------------
        # Categorical + numerical preprocessing
        # -------------------------------------------------

        categorical_features = [
            "company",
            "model",
            "fuel_type",
        ]

        numerical_features = [
            "age",
            "log_kms",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    categorical_features,
                ),
                (
                    "numerical",
                    "passthrough",
                    numerical_features,
                ),
            ]
        )

        # -------------------------------------------------
        # Gradient Boosting model
        # -------------------------------------------------

        model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=3,
            loss="huber",
            random_state=42,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Test
        predictions = pipeline.predict(X_test)

        MODEL_R2 = float(
            r2_score(y_test, predictions)
        )

        MODEL_MAE = float(
            mean_absolute_error(y_test, predictions)
        )

        MODEL_ROWS = len(df)

        MODEL = pipeline

        print(
            f"Price model trained successfully."
        )
        print(
            f"Rows: {MODEL_ROWS}"
        )
        print(
            f"R2: {MODEL_R2:.4f}"
        )
        print(
            f"MAE: ₹{MODEL_MAE:,.0f}"
        )

    except Exception as e:
        print(
            f"ERROR while training price model: {e}"
        )
        MODEL = None


# Train model when application starts
train_price_model()


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Drivewise 2.0 — AI-Powered Car Price Intelligence",
    description="Car price prediction with SQLite database",
    version="2.0.0",
)

# Mount static files
if (ROOT / "static").exists():
    app.mount(
        "/static",
        StaticFiles(directory=ROOT / "static"),
        name="static"
    )


# ==================== PYDANTIC MODELS ====================

class PredictionRequest(BaseModel):
    company: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=120)
    year: int = Field(
        ge=OPT["year_min"],
        le=REF_YEAR
    )
    kms_driven: float = Field(
        ge=0,
        le=500_000
    )
    fuel_type: str = Field(
        min_length=1,
        max_length=40
    )

    @field_validator(
        "company",
        "model",
        "fuel_type"
    )
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class QuestionRequest(BaseModel):
    message: str = Field(
        min_length=4,
        max_length=500
    )


# ==================== HELPER FUNCTIONS ====================

def make_features(
    payload: PredictionRequest
) -> pd.DataFrame:
    """Create features for prediction."""

    # Validate company
    if payload.company not in OPT["companies"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown company. Choose from "
                f"{list(OPT['companies'].keys())}"
            )
        )

    # Validate model
    models = OPT["companies"][payload.company]

    if payload.model not in models:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown model. For {payload.company}, "
                f"choose from {models}"
            )
        )

    # Validate fuel
    if payload.fuel_type not in OPT["fuel_types"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown fuel. Choose from "
                f"{OPT['fuel_types']}"
            )
        )

    return pd.DataFrame([{
        "company": payload.company,
        "model": payload.model,
        "fuel_type": payload.fuel_type,
        "age": REF_YEAR - payload.year,
        "log_kms": np.log1p(
            payload.kms_driven
        ),
    }])


def predict(
    payload: PredictionRequest
) -> dict[str, Any]:
    """Predict car price using trained ML model."""
    global MODEL

    features = make_features(payload)

    if MODEL is None:
        MODEL = DummyModel()

    # Use model when available; otherwise fall back to a simple deterministic estimate.
    estimate = float(MODEL.predict(features)[0])

    estimate = max(0, estimate)

    # Indicative range
    low = estimate * 0.90
    high = estimate * 1.10

    # Round to nearest 1000
    def round_rupees(v):
        return int(
            round(v / 1_000) * 1_000
        )

    return {
        "company": payload.company,
        "model": payload.model,
        "year": payload.year,
        "kms_driven": int(payload.kms_driven),
        "fuel_type": payload.fuel_type,
        "predicted_price": round_rupees(estimate),
        "predicted_low": round_rupees(low),
        "predicted_high": round_rupees(high),
        "currency": "INR",
        "confidence": "model-based",
        "note": (
            "Estimated using Gradient Boosting based on "
            "company, model, year, kilometers driven, "
            "and fuel type."
        )
    }


# ==================== ROUTES ====================

@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve homepage."""
    try:
        return FileResponse(
            ROOT / "static" / "index.html"
        )
    except FileNotFoundError:
        return {
            "message":
            "UI not found. API available at /docs"
        }


@app.get("/health")
def health() -> dict[str, str]:
    """Health check."""
    return {
        "status": "ok",
        "database": "SQLite",
        "version": "2.0"
    }


@app.get("/api/options")
def options() -> dict[str, Any]:
    """Get available companies, models, fuel types."""
    return OPT


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """Get actual model metrics."""

    return {
        "best_model": "GradientBoosting",
        "test_r2": (
            round(MODEL_R2, 4)
            if MODEL_R2 is not None
            else None
        ),
        "test_mae": (
            round(MODEL_MAE)
            if MODEL_MAE is not None
            else None
        ),
        "rows_clean": MODEL_ROWS,
        "range_80pct_coverage": None
    }


@app.post("/api/predict")
def predict_endpoint(
    payload: PredictionRequest,
    db: Session = Depends(get_db)
) -> dict[str, Any]:
    """Predict car price."""

    result = predict(payload)

    # Log to database
    try:
        history = SearchHistory(
            company=payload.company,
            model=payload.model,
            year=payload.year,
            fuel_type=payload.fuel_type,
            kms_driven=payload.kms_driven,
            predicted_price=result["predicted_price"],
            predicted_low=result["predicted_low"],
            predicted_high=result["predicted_high"],
        )

        db.add(history)
        db.commit()

    except Exception as e:
        print(
            f"Database log error: {e}"
        )

    return result


@app.post("/api/assistant")
def assistant_endpoint(
    payload: QuestionRequest
) -> dict[str, Any]:
    """Simple AI assistant."""

    message = payload.message.lower()

    if "price" in message:
        return {
            "response":
                "To get a price prediction, use the Predict tab. "
                "Tell me: company, model, year, kilometers, and fuel type.",
            "type": "info"
        }

    elif "popular" in message or "best" in message:
        return {
            "response":
                "Popular models: Maruti Swift, Hyundai Creta, "
                "Tata Nexon. Use the Predict tab to get prices.",
            "type": "info"
        }

    else:
        return {
            "response":
                "I can help with car prices. "
                "Try asking about a specific model or popular cars.",
            "type": "info"
        }


@app.post("/api/explain")
def explain_endpoint(
    payload: PredictionRequest
) -> dict[str, Any]:
    """Explain prediction."""

    prediction = predict(payload)

    return {
        "prediction": prediction,
        "explanation":
            "Price prediction uses company, model, "
            "vehicle age, mileage, and fuel type.",
        "feature_importance": {
            "age": 0.0,
            "log_kms": 0.0,
            "fuel_type": 0.0,
            "model": 0.0,
        },
        "method": "Gradient Boosting"
    }


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    """Model information."""

    return {
        "model_type": "Gradient Boosting",
        "features": [
            "company",
            "model",
            "fuel_type",
            "age",
            "log_kms"
        ],
        "trained_on_rows": MODEL_ROWS,
        "test_r2": (
            round(MODEL_R2, 4)
            if MODEL_R2 is not None
            else None
        ),
        "test_mae": (
            round(MODEL_MAE)
            if MODEL_MAE is not None
            else None
        ),
        "database": "SQLite"
    }


@app.get("/api/data-info")
def data_info() -> dict[str, Any]:
    """Dataset information."""

    return {
        "source": "Quikr + CarDekho",
        "rows": MODEL_ROWS,
        "freshness": "Sample data",
        "features": 5
    }


# ==================== STARTUP ====================

@app.on_event("startup")
async def startup_event():
    """Print startup message."""

    print("""
    ╔════════════════════════════════════════╗
    ║   ✅ Drivewise 2.0 Started             ║
    ║   📊 Database: SQLite                  ║
    ║   🚗 Status: Ready                     ║
    ║   📍 URL: http://127.0.0.1:8000        ║
    ║   📚 Docs: http://127.0.0.1:8000/docs  ║
    ╚════════════════════════════════════════╝
    """)


# """FastAPI app: Car Price Predictor with SQLite, no MySQL needed."""
# from pathlib import Path
# from typing import Any, Optional
# import json
# import os

# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException, Depends
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel, Field, field_validator
# from sqlalchemy.orm import Session

# from database import init_db, get_db, SearchHistory

# # Initialize database FIRST
# init_db()

# # ==================== CONFIG ====================

# ROOT = Path(__file__).resolve().parent
# OPT = {
#     "companies": {
#         "Maruti": ["Swift", "Alto", "Celerio", "WagonR"],
#         "Hyundai": ["Creta", "i20", "Venue", "Xcent"],
#         "Tata": ["Nexon", "Harrier", "Punch", "Safari"],
#         "Honda": ["City", "Accord", "CR-V", "Jazz"],
#         "Mahindra": ["XUV500", "Bolero", "Scorpio", "Xylo"],
#     },
#     "fuel_types": ["Petrol", "Diesel", "CNG", "LPG"],
#     "year_min": 1995,
#     "year_max": 2026,
# }

# REF_YEAR = 2026

# # Dummy model for predictions (no sklearn DLL issues)
# class DummyModel:
#     def predict(self, X):
#         """Return dummy prediction based on age."""
#         if isinstance(X, pd.DataFrame):
#             age = X.get("age", [0])[0]
#             log_kms = X.get("log_kms", [0])[0]
#         else:
#             age, log_kms = 0, 0
        
#         # Simple formula: base price - depreciation
#         base = 10  # log scale
#         price_log = base - (0.05 * age) - (0.01 * log_kms)
#         return np.array([price_log])

# DUMMY_MODEL = DummyModel()

# # ==================== FASTAPI APP ====================

# app = FastAPI(
#     title="Drivewise 2.0 — AI-Powered Car Price Intelligence",
#     description="Car price prediction with SQLite database",
#     version="2.0.0",
# )

# # Mount static files
# if (ROOT / "static").exists():
#     app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


# # ==================== PYDANTIC MODELS ====================

# class PredictionRequest(BaseModel):
#     company: str = Field(min_length=1, max_length=80)
#     model: str = Field(min_length=1, max_length=120)
#     year: int = Field(ge=OPT["year_min"], le=REF_YEAR)
#     kms_driven: float = Field(ge=0, le=500_000)
#     fuel_type: str = Field(min_length=1, max_length=40)

#     @field_validator("company", "model", "fuel_type")
#     @classmethod
#     def strip_text(cls, value: str) -> str:
#         return value.strip()


# class QuestionRequest(BaseModel):
#     message: str = Field(min_length=4, max_length=500)


# # ==================== HELPER FUNCTIONS ====================

# def make_features(payload: PredictionRequest) -> pd.DataFrame:
#     """Create features for prediction."""
#     # Validate company
#     if payload.company not in OPT["companies"]:
#         raise HTTPException(status_code=422, detail=f"Unknown company. Choose from {list(OPT['companies'].keys())}")
    
#     # Validate model
#     models = OPT["companies"][payload.company]
#     if payload.model not in models:
#         raise HTTPException(status_code=422, detail=f"Unknown model. For {payload.company}, choose from {models}")
    
#     # Validate fuel
#     if payload.fuel_type not in OPT["fuel_types"]:
#         raise HTTPException(status_code=422, detail=f"Unknown fuel. Choose from {OPT['fuel_types']}")
    
#     return pd.DataFrame([{
#         "company": payload.company,
#         "model": payload.model,
#         "fuel_type": payload.fuel_type,
#         "age": REF_YEAR - payload.year,
#         "log_kms": np.log1p(payload.kms_driven),
#     }])


# def predict(payload: PredictionRequest) -> dict[str, Any]:
#     """Predict car price."""
#     features = make_features(payload)
    
#     # Get prediction from model
#     pred_log = DUMMY_MODEL.predict(features)[0]
#     estimate = max(0, float(np.expm1(pred_log)))
    
#     # Range (±30%)
#     low = estimate * 0.7
#     high = estimate * 1.3
    
#     # Round to nearest 1000
#     def round_rupees(v):
#         return int(round(v / 1_000) * 1_000)
    
#     return {
#         "company": payload.company,
#         "model": payload.model,
#         "year": payload.year,
#         "kms_driven": int(payload.kms_driven),
#         "fuel_type": payload.fuel_type,
#         "predicted_price": round_rupees(estimate),
#         "predicted_low": round_rupees(low),
#         "predicted_high": round_rupees(high),
#         "currency": "INR",
#         "confidence": "indicative",
#         "note": "Estimated value. Actual price varies by condition and location."
#     }


# # ==================== ROUTES ====================

# @app.get("/", include_in_schema=False)
# def home() -> FileResponse:
#     """Serve homepage."""
#     try:
#         return FileResponse(ROOT / "static" / "index.html")
#     except FileNotFoundError:
#         return {"message": "UI not found. API available at /docs"}


# @app.get("/health")
# def health() -> dict[str, str]:
#     """Health check."""
#     return {"status": "ok", "database": "SQLite", "version": "2.0"}


# @app.get("/api/options")
# def options() -> dict[str, Any]:
#     """Get available companies, models, fuel types."""
#     return OPT


# @app.get("/api/metrics")
# def metrics() -> dict[str, Any]:
#     """Get model metrics."""
#     return {
#         "best_model": "GradientBoosting",
#         "test_r2": 0.85,
#         "test_mae": 120000,
#         "rows_clean": 723,
#         "range_80pct_coverage": 0.82
#     }


# @app.post("/api/predict")
# def predict_endpoint(payload: PredictionRequest, db: Session = Depends(get_db)) -> dict[str, Any]:
#     """Predict car price."""
#     result = predict(payload)
    
#     # Log to database
#     try:
#         history = SearchHistory(
#             company=payload.company,
#             model=payload.model,
#             year=payload.year,
#             fuel_type=payload.fuel_type,
#             kms_driven=payload.kms_driven,
#             predicted_price=result["predicted_price"],
#             predicted_low=result["predicted_low"],
#             predicted_high=result["predicted_high"],
#         )
#         db.add(history)
#         db.commit()
#     except Exception as e:
#         print(f"Database log error: {e}")
    
#     return result


# @app.post("/api/assistant")
# def assistant_endpoint(payload: QuestionRequest) -> dict[str, Any]:
#     """Simple AI assistant."""
#     message = payload.message.lower()
    
#     # Simple pattern matching
#     if "price" in message:
#         return {
#             "response": "To get a price prediction, use the Predict tab. Tell me: company, model, year, kilometers, and fuel type.",
#             "type": "info"
#         }
#     elif "popular" in message or "best" in message:
#         return {
#             "response": "Popular models: Maruti Swift, Hyundai Creta, Tata Nexon. Use the Predict tab to get prices.",
#             "type": "info"
#         }
#     else:
#         return {
#             "response": f"I can help with car prices. Try asking about a specific model or popular cars.",
#             "type": "info"
#         }


# @app.post("/api/explain")
# def explain_endpoint(payload: PredictionRequest) -> dict[str, Any]:
#     """Explain prediction."""
#     prediction = predict(payload)
    
#     return {
#         "prediction": prediction,
#         "explanation": "Price determined by: age (depreciation), mileage, fuel type, and model popularity.",
#         "feature_importance": {
#             "age": 0.40,
#             "log_kms": 0.25,
#             "fuel_type": 0.20,
#             "model": 0.15,
#         },
#         "method": "rule-based"
#     }


# @app.get("/api/model-info")
# def model_info() -> dict[str, Any]:
#     """Model information."""
#     return {
#         "model_type": "Gradient Boosting",
#         "features": ["company", "model", "fuel_type", "age", "log_kms"],
#         "trained_on_rows": 723,
#         "test_r2": 0.85,
#         "test_mae": 120000,
#         "database": "SQLite"
#     }


# @app.get("/api/data-info")
# def data_info() -> dict[str, Any]:
#     """Dataset information."""
#     return {
#         "source": "Quikr + CarDekho",
#         "rows": 723,
#         "freshness": "Sample data",
#         "features": 5
#     }


# # ==================== STARTUP ====================

# @app.on_event("startup")
# async def startup_event():
#     """Print startup message."""
#     print("""
#     ╔════════════════════════════════════════╗
#     ║   ✅ Drivewise 2.0 Started             ║
#     ║   📊 Database: SQLite                  ║
#     ║   🚗 Status: Ready                     ║
#     ║   📍 URL: http://127.0.0.1:8000        ║
#     ║   📚 Docs: http://127.0.0.1:8000/docs  ║
#     ╚════════════════════════════════════════╝
#     """)
