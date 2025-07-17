import hashlib
import hmac
from typing import Optional
from loguru import logger

class SecurityManager:
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format"""
        if not api_key or len(api_key) < 8:
            logger.error("Invalid API key format")
            return False
        return True
    
    @staticmethod
    def generate_request_signature(data: str, secret: str) -> str:
        """Generate HMAC signature for request validation"""
        return hmac.new(
            secret.encode('utf-8'),
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> Optional[str]:
        """Sanitize and validate stock symbol"""
        if not symbol:
            return None
            
        cleaned = ''.join(c.upper() for c in symbol if c.isalnum())
        
        if not (1 <= len(cleaned) <= 5):
            logger.warning(f"Invalid symbol format: {symbol}")
            return None
            
        return cleaned 