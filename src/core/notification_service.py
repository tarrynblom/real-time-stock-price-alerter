from typing import List, Dict, Any, Protocol
from abc import abstractmethod
from loguru import logger
from src.core.alerting_engine import Alert, AlertSeverity


class NotificationChannel(Protocol):
    """Protocol for notification channels"""
    
    @abstractmethod
    def send_notification(self, alert: Alert) -> bool:
        """Send notification for an alert"""
        pass


class ConsoleNotifier:
    """Console-based notification channel"""
    
    def send_notification(self, alert: Alert) -> bool:
        """Print alert to console with formatting"""
        try:
            severity_emoji = {
                AlertSeverity.LOW: "ℹ️",
                AlertSeverity.MEDIUM: "⚠️", 
                AlertSeverity.HIGH: "🚨",
                AlertSeverity.CRITICAL: "🔥"
            }
            
            emoji = severity_emoji.get(alert.severity, "📢")
            
            print(f"\n{emoji} STOCK ALERT - {alert.severity.value.upper()}")
            print(f"Symbol: {alert.symbol}")
            print(f"Type: {alert.alert_type.value.replace('_', ' ').title()}")
            print(f"Message: {alert.message}")
            print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 50)
            
            return True
            
        except Exception as e:
            logger.error(f"Console notification failed: {e}")
            return False


class LogFileNotifier:
    """Log file notification channel"""
    
    def __init__(self, log_file: str = "logs/alerts.log"):
        self.log_file = log_file
    
    def send_notification(self, alert: Alert) -> bool:
        """Write alert to log file"""
        try:
            log_entry = (
                f"[{alert.timestamp.isoformat()}] "
                f"ALERT|{alert.severity.value.upper()}|{alert.symbol}|"
                f"{alert.alert_type.value}|{alert.message}"
            )
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry + '\n')
            
            return True
            
        except Exception as e:
            logger.error(f"Log file notification failed: {e}")
            return False


class NotificationService:
    """Orchestrates multiple notification channels"""
    
    def __init__(self):
        self.channels: List[NotificationChannel] = [
            ConsoleNotifier(),
            LogFileNotifier()
        ]
    
    def send_alerts(self, alerts: List[Alert]) -> Dict[str, Any]:
        """Send alerts through all configured channels"""
        results = {
            'total_alerts': len(alerts),
            'successful_notifications': 0,
            'failed_notifications': 0,
            'channels_used': len(self.channels)
        }
        
        for alert in alerts:
            for channel in self.channels:
                try:
                    success = channel.send_notification(alert)
                    if success:
                        results['successful_notifications'] += 1
                    else:
                        results['failed_notifications'] += 1
                except Exception as e:
                    logger.error(f"Notification channel failed: {e}")
                    results['failed_notifications'] += 1
        
        return results
    
    def add_channel(self, channel: NotificationChannel) -> None:
        """Add a new notification channel"""
        self.channels.append(channel)
        logger.info(f"Added notification channel: {channel.__class__.__name__}") 