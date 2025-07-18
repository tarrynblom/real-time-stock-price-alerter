# src/core/api_client.py
from typing import Dict, Any, Optional
import requests
import time
from loguru import logger
from config.settings import settings


class APIClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "StockAlerter/1.0", "Accept": "application/json"}
        )
        self.last_request_time = 0
        self.rate_limit_delay = 12  # 5 calls per minute = 12 seconds between calls

    def _rate_limit(self):
        """Implement intelligent rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            logger.info(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _make_request(
        self, url: str, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Make authenticated API request with error handling"""
        self._rate_limit()

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API-specific errors
            if "Error Message" in data:
                logger.error(f"API Error: {data['Error Message']}")
                return None

            if "Note" in data:
                logger.warning(f"API Note: {data['Note']}")
                return None

            return data

        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid JSON response: {e}")
            return None


class AlphaVantageClient(APIClient):
    def __init__(self):
        super().__init__()
        self.api_key = settings.alpha_vantage_api_key
        self.base_url = settings.api_base_url

    def get_intraday_data(
        self, symbol: str, interval: str = "5min"
    ) -> Optional[Dict[str, Any]]:
        """Fetch intraday stock data"""
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "outputsize": "compact",
        }

        logger.info(f"Fetching intraday data for {symbol}")
        return self._make_request(self.base_url, params)
