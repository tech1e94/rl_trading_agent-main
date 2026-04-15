from typing import Annotated, Dict, List, Literal

import os
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from stable_baselines3 import DQN, PPO

ACTIONS = [
    "EXIT", "HOLD", "AGGRESSIVE_LONG",
    "CONSERVATIVE_LONG", "TREND_FOLLOW", "MEAN_REVERT"
]
HOLDING_DAYS = [1, 3, 5, 10]
MODEL_PATHS = {
    "sentiment": "models/saved/sentiment_agent.zip",
}

models: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        models["sentiment"] = load_model(MODEL_PATHS["sentiment"])
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load sentiment model from {MODEL_PATHS['sentiment']}: {exc}"
        ) from exc
    yield


app = FastAPI(
    title="RL Trading Agent API",
    description="API for serving prediction results from the sentiment trading agent.",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


class PredictRequest(BaseModel):
    state: Annotated[List[float], Field(min_length=30, max_length=30)]


class PredictResponse(BaseModel):
    model: str
    action_id: int
    strategy: str
    holding_days: int
    position: str
    portfolio_signal: str
    expected_return_pct: float
    estimated_portfolio_value: float
    profit_loss_status: str


class CompareResponse(BaseModel):
    predictions: Dict[str, PredictResponse]


def load_model(path: str):
    return PPO.load(path)


def compute_portfolio_metrics(state: List[float], action_id: int) -> Dict[str, object]:
    sentiment_features = state[14:]
    sentiment_avg = float(np.mean(sentiment_features)) if sentiment_features else 0.0
    trend_strength = float(state[12]) if len(state) > 12 else 0.0
    regime = float(state[13]) if len(state) > 13 else 0.0
    current_price = float(np.mean([state[4], state[5], state[9]])) if len(state) >= 10 else 100.0

    action = ACTIONS[action_id % len(ACTIONS)]
    holding_days = HOLDING_DAYS[action_id // len(ACTIONS)]

    if action == "EXIT":
        expected_pct = -0.002 * holding_days
        position = "Flat / Close Position"
        portfolio_signal = "Exit position"
    elif action == "HOLD":
        expected_pct = 0.0
        position = "Hold Current Position"
        portfolio_signal = "Maintain position"
    elif action == "AGGRESSIVE_LONG":
        expected_pct = 0.012 * holding_days + 0.03 * sentiment_avg + 0.005 * trend_strength
        position = "Aggressive Long"
        portfolio_signal = "Bullish momentum"
    elif action == "CONSERVATIVE_LONG":
        expected_pct = 0.006 * holding_days + 0.02 * sentiment_avg + 0.003 * trend_strength
        position = "Conservative Long"
        portfolio_signal = "Cautious bullish"
    elif action == "TREND_FOLLOW":
        expected_pct = 0.008 * holding_days + 0.025 * sentiment_avg + 0.006 * trend_strength
        position = "Trend Following Long"
        portfolio_signal = "Trend-driven bullish"
    else:  # MEAN_REVERT
        expected_pct = 0.004 * holding_days - 0.02 * sentiment_avg + 0.002 * regime
        position = "Mean Reversion"
        portfolio_signal = "Reversion trade"

    expected_pct = max(-0.15, min(expected_pct, 0.15))
    estimated_portfolio_value = round(10000.0 * (1 + expected_pct), 2)

    if expected_pct > 0.0:
        profit_loss_status = "Projected Profit"
    elif expected_pct < 0.0:
        profit_loss_status = "Projected Loss"
    else:
        profit_loss_status = "Neutral / No expected change"

    return {
        "position": position,
        "portfolio_signal": portfolio_signal,
        "expected_return_pct": round(expected_pct * 100, 2),
        "estimated_portfolio_value": estimated_portfolio_value,
        "profit_loss_status": profit_loss_status,
    }


def predict_from_model(model, state: List[float]) -> PredictResponse:
    state_array = np.array(state, dtype=np.float32).reshape(1, 30)
    action, _ = model.predict(state_array)
    action_id = int(action[0])

    # Validate action_id (should be 0-23: 6 actions * 4 holding days)
    if action_id < 0 or action_id >= len(ACTIONS) * len(HOLDING_DAYS):
        # Clamp to valid range
        action_id = max(0, min(action_id, len(ACTIONS) * len(HOLDING_DAYS) - 1))

    metrics = compute_portfolio_metrics(state, action_id)

    return PredictResponse(
        model="sentiment",
        action_id=action_id,
        strategy=ACTIONS[action_id % len(ACTIONS)],
        holding_days=HOLDING_DAYS[action_id // len(ACTIONS)],
        position=metrics["position"],
        portfolio_signal=metrics["portfolio_signal"],
        expected_return_pct=metrics["expected_return_pct"],
        estimated_portfolio_value=metrics["estimated_portfolio_value"],
        profit_loss_status=metrics["profit_loss_status"],
    )


@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "ok",
        "model_loaded": "sentiment" in models,
        "model_type": type(models.get("sentiment")).__name__ if "sentiment" in models else None
    }


@app.get("/data/sample", tags=["Data"])
def get_sample_data():
    """Return sample news and market data used in training"""
    import csv
    
    sample_news = []
    sample_sentiment = []
    
    # Load sample news
    try:
        with open("data/raw/news/AAPL_news.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:  # First 5 news items
                    break
                sample_news.append({
                    "date": row.get("date"),
                    "headline": row.get("headline"),
                    "category": row.get("category"),
                    "source": row.get("source")
                })
    except Exception as e:
        sample_news = [{"error": f"Could not load news: {str(e)}"}]
    
    # Load sample sentiment data
    try:
        with open("data/raw/news/AAPL_sector_scored.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:  # First 5 sentiment items
                    break
                sample_sentiment.append({
                    "date": row.get("date"),
                    "headline": row.get("headline"),
                    "sentiment_score": float(row.get("sentiment_score", 0)),
                    "relevance": row.get("relevance_level"),
                    "ticker": row.get("fetched_for_ticker")
                })
    except Exception as e:
        sample_sentiment = [{"error": f"Could not load sentiment: {str(e)}"}]
    
    return {
        "news_headlines": sample_news,
        "sentiment_analysis": sample_sentiment
    }


@app.get("/test-vectors", tags=["Testing"])
def get_test_vectors():
    """Return multiple test vectors for different market scenarios"""
    
    test_vectors = {
        "neutral_market": {
            "name": "Neutral Market",
            "description": "Balanced technical indicators, neutral sentiment",
            "state": [50.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.02, 0.02, 100.0, 100.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "bullish_market": {
            "name": "Bullish Market",
            "description": "High RSI, positive sentiment, uptrend indicators",
            "state": [75.0, 1.5, 0.5, 1.0, 120.0, 110.0, 0.015, 0.018, 110.0, 105.0, 100.0, 0.3, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.0, 0.0]
        },
        "bearish_market": {
            "name": "Bearish Market",
            "description": "Low RSI, negative sentiment, downtrend indicators",
            "state": [25.0, -1.5, -0.5, -1.0, 80.0, 90.0, 0.035, 0.045, 90.0, 95.0, 100.0, 1.2, -1.5, -1.0, -0.5, -0.5, -0.5, -0.5, -0.5, -0.3, -0.3, -0.2, -0.2, -0.1, -0.1, -0.1, -0.05, -0.05, 0.0, 0.0]
        },
        "high_volatility": {
            "name": "High Volatility",
            "description": "High volatility, mixed sentiment, extreme technical values",
            "state": [55.0, 2.0, 1.0, 2.5, 130.0, 85.0, 0.08, 0.12, 120.0, 100.0, 80.0, 2.5, 0.5, -0.5, 0.8, 0.8, -0.5, 0.5, -0.3, 0.3, 0.0, -0.2, 0.2, -0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        },
        "positive_sentiment": {
            "name": "Positive Sentiment Only",
            "description": "Neutral technicals, strong positive news sentiment",
            "state": [50.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.02, 0.02, 100.0, 100.0, 100.0, 0.5, 0.0, 0.0, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6, 0.6, 0.5, 0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2]
        },
        "negative_sentiment": {
            "name": "Negative Sentiment Only",
            "description": "Neutral technicals, strong negative news sentiment",
            "state": [50.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.02, 0.02, 100.0, 100.0, 100.0, 0.5, 0.0, 0.0, -0.8, -0.8, -0.7, -0.7, -0.6, -0.6, -0.6, -0.5, -0.5, -0.5, -0.4, -0.4, -0.3, -0.3, -0.2, -0.2]
        }
    }
    
    return test_vectors
    
    sample_news = []
    sample_sentiment = []
    
    # Load sample news
    try:
        with open("data/raw/news/AAPL_news.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:  # First 5 news items
                    break
                sample_news.append({
                    "date": row.get("date"),
                    "headline": row.get("headline"),
                    "category": row.get("category"),
                    "source": row.get("source")
                })
    except Exception as e:
        sample_news = [{"error": f"Could not load news: {str(e)}"}]
    
    # Load sample sentiment data
    try:
        with open("data/raw/news/AAPL_sector_scored.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 5:  # First 5 sentiment items
                    break
                sample_sentiment.append({
                    "date": row.get("date"),
                    "headline": row.get("headline"),
                    "sentiment_score": float(row.get("sentiment_score", 0)),
                    "relevance": row.get("relevance_level"),
                    "ticker": row.get("fetched_for_ticker")
                })
    except Exception as e:
        sample_sentiment = [{"error": f"Could not load sentiment: {str(e)}"}]
    
    return {
        "news_headlines": sample_news,
        "sentiment_analysis": sample_sentiment
    }


@app.get("/web", tags=["Web UI"])
def web_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/web/test", tags=["Web UI"])
def web_test(request: Request):
    try:
        # Sample state: 30 features (technical + sentiment)
        # Example: RSI=50, MACD=0, etc., sentiment scores around 0
        sample_state = [
            50.0,  # RSI
            0.0,   # MACD
            0.0,   # MACD signal
            0.0,   # MACD hist
            100.0, # SMA short
            100.0, # SMA long
            0.02,  # Volatility short
            0.02,  # Volatility long
            100.0, # BB upper
            100.0, # BB middle
            100.0, # BB lower
            0.5,   # ATR
            0.0,   # Trend strength
            0.0,   # Regime
        ] + [0.0] * 16  # Sentiment features (16 sentiment scores)
        
        prediction = predict_from_model(models["sentiment"], sample_state)
        prediction.model = "sentiment"
        
        result = {
            "status": "success",
            "message": "Model test successful",
            "sample_input": sample_state,
            "prediction": prediction.model_dump()
        }
        
        # For debugging, return JSON instead of template
        return result
        
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@app.post("/web/predict", tags=["Web UI"])
def web_predict(request: Request, model: str = Form(...), state: str = Form(...)):
    try:
        state_list = [float(x.strip()) for x in state.split(",")]
        if len(state_list) != 30:
            raise ValueError(f"Expected 30 features, got {len(state_list)}")
        
        prediction = predict_from_model(models["sentiment"], state_list)
        prediction.model = "sentiment"
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "status": "success",
                "model": model,
                "input_state": state_list,
                "prediction": prediction.dict()
            }
        })
    except Exception as exc:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "status": "error",
                "message": str(exc)
            }
        })


@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
def predict(request: PredictRequest):
    """JSON API endpoint for predictions from frontend"""
    try:
        prediction = predict_from_model(models["sentiment"], request.state)
        prediction.model = "sentiment"
        return prediction
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/test", tags=["Testing"])
def test_model():
    """
    Run a simple test prediction on the sentiment model.
    Returns the prediction result for demonstration.
    """
    try:
        # Sample state: 30 features (technical + sentiment)
        # Example: RSI=50, MACD=0, etc., sentiment scores around 0
        sample_state = [
            50.0,  # RSI
            0.0,   # MACD
            0.0,   # MACD signal
            0.0,   # MACD hist
            100.0, # SMA short
            100.0, # SMA long
            0.02,  # Volatility short
            0.02,  # Volatility long
            100.0, # BB upper
            100.0, # BB middle
            100.0, # BB lower
            0.5,   # ATR
            0.0,   # Trend strength
            0.0,   # Regime
        ] + [0.0] * 16  # Sentiment features (16 sentiment scores)
        
        prediction = predict_from_model(models["sentiment"], sample_state)
        prediction.model = "sentiment"
        
        return {
            "status": "success",
            "message": "Model test successful",
            "sample_input": sample_state,
            "prediction": prediction.model_dump()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
