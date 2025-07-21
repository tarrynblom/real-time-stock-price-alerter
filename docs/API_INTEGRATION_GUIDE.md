# 🔌 API Integration Guide

Complete guide for integrating with the Stock Price Alerter API, including authentication, error handling, and real-world usage patterns.

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Real-World Examples](#real-world-examples)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Best Practices](#best-practices)
- [SDKs & Libraries](#sdks--libraries)

---

## 🚀 Quick Start

### Base URL
```
http://localhost:8000
```

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "model_trained": true,
  "version": "1.0.0",
  "services": {
    "api": {"status": "healthy"}
  },
  "metrics": {
    "models_loaded": 1
  }
}
```

---

## 🔐 Authentication

Currently, the API uses API key authentication for external data sources. No authentication is required for API endpoints in the demo version.

**Production Considerations:**
- Add JWT tokens for API access
- Implement rate limiting per user
- Add API key authentication for endpoints

---

## 📡 API Endpoints

### 1. Health Check
**GET** `/health`

Monitor API status and model training state.

```bash
curl -X GET "http://localhost:8000/health"
```

### 2. Train Model
**POST** `/train`

Train the ML model for a specific stock symbol.

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "interval": "5min"
  }'
```

**Response:**
```json
{
  "message": "Model training started for AAPL",
  "symbol": "AAPL",
  "status": "training_in_progress"
}
```

### 3. Get Prediction
**POST** `/predict`

Get price prediction for a stock symbol.

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "interval": "5min"
  }'
```

**Response:**
```json
{
  "symbol": "AAPL",
  "current_price": 150.25,
  "predicted_price": 151.30,
  "price_change": 1.05,
  "price_change_pct": 0.70,
  "prediction_time": "2024-01-15T10:30:00Z",
  "confidence": "medium"
}
```

### 4. Check Alerts
**POST** `/alert`

Get predictions and trigger alerts if thresholds are met.

```bash
curl -X POST "http://localhost:8000/alert" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "interval": "5min"
  }'
```

**Response:**
```json
{
  "success": true,
  "symbol": "AAPL",
  "prediction": {
    "symbol": "AAPL",
    "current_price": 150.25,
    "predicted_price": 151.30,
    "price_change": 1.05,
    "price_change_pct": 0.70,
    "prediction_time": "2024-01-15T10:30:00Z",
    "confidence": "medium"
  },
  "alerts_triggered": 0,
  "alerts": [],
  "notification_results": {
    "total_alerts": 0,
    "successful_notifications": 0,
    "failed_notifications": 0,
    "channels_used": 2
  }
}
```

---

## 💼 Real-World Examples

### Portfolio Monitoring System

```python
import requests
import time
from typing import List, Dict

class StockAlerter:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
    
    def train_portfolio(self, symbols: List[str]) -> Dict:
        """Train models for multiple stocks"""
        results = {}
        for symbol in symbols:
            try:
                response = self.session.post(
                    f"{self.base_url}/train",
                    json={"symbol": symbol, "interval": "5min"}
                )
                response.raise_for_status()
                results[symbol] = response.json()
                
                # Respect rate limits
                time.sleep(2)
                
            except requests.RequestException as e:
                results[symbol] = {"error": str(e)}
        
        return results
    
    def monitor_portfolio(self, symbols: List[str]) -> Dict:
        """Monitor portfolio and get alerts"""
        alerts = {}
        for symbol in symbols:
            try:
                response = self.session.post(
                    f"{self.base_url}/alert",
                    json={"symbol": symbol, "interval": "5min"}
                )
                response.raise_for_status()
                result = response.json()
                
                if result["alerts_triggered"] > 0:
                    alerts[symbol] = result["alerts"]
                
                time.sleep(1)  # Rate limiting
                
            except requests.RequestException as e:
                print(f"Error monitoring {symbol}: {e}")
        
        return alerts

# Usage Example
alerter = StockAlerter()

# Portfolio setup
portfolio = ["AAPL", "GOOGL", "MSFT", "TSLA"]

# Train models
print("Training models...")
training_results = alerter.train_portfolio(portfolio)

# Wait for training to complete
time.sleep(30)

# Monitor for alerts
print("Monitoring portfolio...")
while True:
    alerts = alerter.monitor_portfolio(portfolio)
    
    if alerts:
        print(f"🚨 Alerts triggered: {alerts}")
    else:
        print("✅ No alerts - portfolio stable")
    
    time.sleep(300)  # Check every 5 minutes
```

### Webhook Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
ALERTER_API = "http://localhost:8000"

@app.route('/webhook/stock_alert', methods=['POST'])
def handle_stock_webhook():
    """Handle incoming webhook for stock monitoring"""
    data = request.json
    symbol = data.get('symbol')
    
    if not symbol:
        return jsonify({"error": "Symbol required"}), 400
    
    try:
        # Get prediction and alerts
        response = requests.post(
            f"{ALERTER_API}/alert",
            json={"symbol": symbol, "interval": "5min"},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Forward to external systems if alerts triggered
        if result["alerts_triggered"] > 0:
            # Send to Slack, Discord, email, etc.
            notify_external_systems(result)
        
        return jsonify(result)
        
    except requests.RequestException as e:
        return jsonify({"error": str(e)}), 500

def notify_external_systems(alert_data):
    """Send alerts to external notification systems"""
    # Example: Slack webhook
    slack_webhook = "https://hooks.slack.com/your/webhook/url"
    
    for alert in alert_data["alerts"]:
        message = {
            "text": f"🚨 Stock Alert: {alert['message']}",
            "attachments": [{
                "color": "danger" if alert["severity"] == "high" else "warning",
                "fields": [
                    {"title": "Symbol", "value": alert_data["symbol"], "short": True},
                    {"title": "Type", "value": alert["type"], "short": True},
                    {"title": "Severity", "value": alert["severity"], "short": True}
                ]
            }]
        }
        
        requests.post(slack_webhook, json=message)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### JavaScript/Node.js Client

```javascript
class StockAlerterClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }

    async makeRequest(endpoint, method = 'GET', data = null) {
        const config = {
            method,
            headers: {
                'Content-Type': 'application/json',
            },
        };

        if (data) {
            config.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, config);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    async healthCheck() {
        return this.makeRequest('/health');
    }

    async trainModel(symbol, interval = '5min') {
        return this.makeRequest('/train', 'POST', { symbol, interval });
    }

    async getPrediction(symbol, interval = '5min') {
        return this.makeRequest('/predict', 'POST', { symbol, interval });
    }

    async checkAlerts(symbol, interval = '5min') {
        return this.makeRequest('/alert', 'POST', { symbol, interval });
    }

    async monitorSymbol(symbol, callback, intervalMs = 60000) {
        const monitor = async () => {
            try {
                const result = await this.checkAlerts(symbol);
                callback(null, result);
            } catch (error) {
                callback(error, null);
            }
        };

        // Initial check
        await monitor();

        // Set up interval monitoring
        return setInterval(monitor, intervalMs);
    }
}

// Usage Example
const client = new StockAlerterClient();

// Monitor Apple stock
client.monitorSymbol('AAPL', (error, result) => {
    if (error) {
        console.error('Monitoring error:', error);
        return;
    }

    if (result.alerts_triggered > 0) {
        console.log('🚨 Alerts triggered:', result.alerts);
        
        // Send browser notification
        if ('Notification' in window) {
            new Notification('Stock Alert!', {
                body: result.alerts[0].message,
                icon: '/stock-icon.png'
            });
        }
    }
}, 30000); // Check every 30 seconds
```

---

## ⚠️ Error Handling

### Common Error Responses

#### 400 Bad Request
```json
{
  "detail": "Model not trained. Call /train endpoint first."
}
```

#### 422 Validation Error
```json
{
  "detail": [
    {
      "loc": ["body", "symbol"],
      "msg": "ensure this value has at most 5 characters",
      "type": "value_error.any_str.max_length",
      "ctx": {"limit_value": 5}
    }
  ]
}
```

#### 500 Internal Server Error
```json
{
  "detail": "Failed to fetch data for INVALID_SYMBOL"
}
```

### Error Handling Best Practices

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_robust_session():
    """Create a session with retry logic and timeouts"""
    session = requests.Session()
    
    # Retry strategy
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "OPTIONS", "POST"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def safe_api_call(session, url, json_data, timeout=30):
    """Make API call with comprehensive error handling"""
    try:
        response = session.post(url, json=json_data, timeout=timeout)
        response.raise_for_status()
        return response.json(), None
        
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except requests.exceptions.ConnectionError:
        return None, "Connection error - check if API is running"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            return None, f"Bad request: {e.response.json().get('detail', 'Unknown error')}"
        elif e.response.status_code == 422:
            errors = e.response.json().get('detail', [])
            return None, f"Validation error: {errors}"
        else:
            return None, f"HTTP {e.response.status_code}: {e.response.text}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

# Usage
session = create_robust_session()
result, error = safe_api_call(
    session, 
    "http://localhost:8000/predict", 
    {"symbol": "AAPL", "interval": "5min"}
)

if error:
    print(f"Error: {error}")
else:
    print(f"Prediction: {result}")
```

---

## 🚦 Rate Limiting

### Current Limitations
- **Alpha Vantage API**: 5 calls per minute, 500 calls per day
- **Internal Processing**: No artificial limits (designed for high throughput)

### Best Practices
- **Space out requests**: Wait 12+ seconds between API calls
- **Batch operations**: Train multiple models sequentially with delays
- **Cache results**: Store predictions temporarily to reduce API calls
- **Use webhooks**: For real-time monitoring instead of polling

### Rate Limit Handling

```python
import time
from datetime import datetime, timedelta
from collections import defaultdict

class RateLimiter:
    def __init__(self, calls_per_minute=5):
        self.calls_per_minute = calls_per_minute
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if we're approaching rate limits"""
        now = datetime.now()
        
        # Remove calls older than 1 minute
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(minutes=1)]
        
        if len(self.calls) >= self.calls_per_minute:
            oldest_call = min(self.calls)
            wait_time = 60 - (now - oldest_call).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        self.calls.append(now)

# Usage
rate_limiter = RateLimiter(calls_per_minute=5)

for symbol in ["AAPL", "GOOGL", "MSFT"]:
    rate_limiter.wait_if_needed()
    # Make API call
    response = requests.post("http://localhost:8000/train", 
                           json={"symbol": symbol})
```

---

## ✅ Best Practices

### 1. Production Deployment
```python
# Use environment variables
import os

API_BASE_URL = os.getenv('STOCK_ALERTER_API_URL', 'http://localhost:8000')
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))
RETRY_ATTEMPTS = int(os.getenv('RETRY_ATTEMPTS', '3'))
```

### 2. Async Operations
```python
import asyncio
import aiohttp

async def async_predict(session, symbol):
    """Async prediction request"""
    async with session.post(
        f"{API_BASE_URL}/predict",
        json={"symbol": symbol, "interval": "5min"}
    ) as response:
        return await response.json()

async def monitor_portfolio_async(symbols):
    """Monitor multiple stocks concurrently"""
    async with aiohttp.ClientSession() as session:
        tasks = [async_predict(session, symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return dict(zip(symbols, results))

# Usage
symbols = ["AAPL", "GOOGL", "MSFT"]
results = asyncio.run(monitor_portfolio_async(symbols))
```

### 3. Data Validation
```python
from pydantic import BaseModel, validator

class StockSymbol(BaseModel):
    symbol: str
    interval: str = "5min"
    
    @validator('symbol')
    def validate_symbol(cls, v):
        if not v or len(v) > 5:
            raise ValueError('Symbol must be 1-5 characters')
        return v.upper().strip()
    
    @validator('interval')
    def validate_interval(cls, v):
        valid_intervals = ['1min', '5min', '15min', '30min', '60min']
        if v not in valid_intervals:
            raise ValueError(f'Interval must be one of {valid_intervals}')
        return v

# Usage
try:
    request = StockSymbol(symbol="aapl", interval="5min")
    # Make API call with validated data
except ValueError as e:
    print(f"Validation error: {e}")
```

### 4. Logging and Monitoring
```python
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_alerter_client.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_api_call(method, endpoint, data, response_time, status_code):
    """Log API call details for monitoring"""
    logger.info(f"API Call: {method} {endpoint} - "
               f"Status: {status_code} - "
               f"Response Time: {response_time:.2f}s - "
               f"Data: {data}")

# Usage in API calls
start_time = time.time()
response = requests.post(url, json=data)
response_time = time.time() - start_time

log_api_call("POST", "/predict", data, response_time, response.status_code)
```

---

## 📦 SDKs & Libraries

### Python SDK Structure

```python
# stock_alerter_sdk/client.py
from typing import Dict, List, Optional
import requests
from dataclasses import dataclass

@dataclass
class PredictionResult:
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_pct: float
    confidence: str

@dataclass
class Alert:
    type: str
    severity: str
    message: str
    timestamp: str

class StockAlerterSDK:
    """Official Python SDK for Stock Alerter API"""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def train(self, symbol: str, interval: str = "5min") -> Dict:
        """Train model for symbol"""
        response = self.session.post(
            f"{self.base_url}/train",
            json={"symbol": symbol, "interval": interval},
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def predict(self, symbol: str, interval: str = "5min") -> PredictionResult:
        """Get prediction for symbol"""
        response = self.session.post(
            f"{self.base_url}/predict",
            json={"symbol": symbol, "interval": interval},
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return PredictionResult(**data)
    
    def get_alerts(self, symbol: str, interval: str = "5min") -> List[Alert]:
        """Get alerts for symbol"""
        response = self.session.post(
            f"{self.base_url}/alert",
            json={"symbol": symbol, "interval": interval},
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        return [Alert(**alert) for alert in data.get("alerts", [])]

# Installation: pip install stock-alerter-sdk
# Usage:
# from stock_alerter_sdk import StockAlerterSDK
# 
# client = StockAlerterSDK("http://localhost:8000")
# prediction = client.predict("AAPL")
# print(f"AAPL prediction: ${prediction.predicted_price:.2f}")
```

### CLI Tool

```bash
# Install CLI tool
pip install stock-alerter-cli

# Usage examples
stock-alerter health
stock-alerter train AAPL
stock-alerter predict AAPL
stock-alerter monitor AAPL GOOGL MSFT --interval 5min
stock-alerter alerts AAPL --watch
```

---

## 🔗 Related Resources

- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI
- **[Deployment Guide](./DEPLOYMENT_TROUBLESHOOTING.md)** - Setup and troubleshooting
- **[Demo Materials](./demo/)** - Live demonstration scripts
- **[Source Code](https://github.com/your-username/real-time-stock-price-alerter)** - Full implementation

---

**📧 Support**: For integration help or feature requests, please open an issue on GitHub or contact the development team. 