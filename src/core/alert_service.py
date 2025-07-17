from typing import Dict, Any, List
from loguru import logger
from src.core.alerting_engine import AlertingEngine, Alert
from src.core.notification_service import NotificationService
from src.core.prediction_service import PredictionService


class AlertService:
    """Complete alerting service that integrates prediction and notification"""
    
    def __init__(self, prediction_service: PredictionService):
        self.prediction_service = prediction_service
        self.alerting_engine = AlertingEngine()
        self.notification_service = NotificationService()
    
    def check_and_alert(self, symbol: str, interval: str = "5min") -> Dict[str, Any]:
        """Complete workflow: predict -> evaluate -> alert"""
        try:
            # 1. Get prediction
            prediction_result = self.prediction_service.predict_next_price(symbol, interval)
            
            if not prediction_result.get('prediction_successful', True):
                return {
                    'success': False,
                    'error': prediction_result.get('error', 'Prediction failed')
                }
            
            # 2. Evaluate for alerts
            alerts = self.alerting_engine.evaluate_prediction(prediction_result)
            
            # 3. Send notifications
            notification_results = {}
            if alerts:
                notification_results = self.notification_service.send_alerts(alerts)
                logger.info(f"Generated {len(alerts)} alerts for {symbol}")
            else:
                logger.info(f"No alerts triggered for {symbol}")
            
            return {
                'success': True,
                'symbol': symbol,
                'prediction': prediction_result,
                'alerts_triggered': len(alerts),
                'alerts': [
                    {
                        'type': alert.alert_type.value,
                        'severity': alert.severity.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in alerts
                ],
                'notification_results': notification_results
            }
            
        except Exception as e:
            logger.error(f"Alert service failed for {symbol}: {e}")
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            } 