# RL Trading Agent Frontend

A minimal web interface for demonstrating the sentiment-based RL trading model.

## Setup

1. Deploy the backend API (see main project README)
2. Update `script.js` with your deployed API URL:
   ```javascript
   const API_BASE_URL = 'https://your-api-url.railway.app';
   ```
3. Serve the frontend files (HTML, CSS, JS) on any static host (GitHub Pages, Netlify, etc.)

## Features

- **Test Model**: Run prediction with sample market data
- **Custom Prediction**: Input your own 30-feature state vector
- **Real-time Results**: See model decisions instantly

## API Endpoints Used

- `GET /test` - Test with sample data
- `POST /predict` - Custom prediction with state vector

## State Vector Format

30 comma-separated floats representing:
- Technical indicators (RSI, MACD, etc.)
- Sentiment scores

Example: `50.0, 0.0, 0.0, 0.0, 100.0, 100.0, 0.02, 0.02, 100.0, 100.0, 100.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0`