from typing import Dict, Any, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from loguru import logger


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    PRICE_INCREASE = "price_increase"
    PRICE_DECREASE = "price_decrease"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_ANOMALY = "volume_anomaly"


@dataclass
class AlertRule:
    """Configuration for alert triggers"""

    threshold_pct: float
    severity: AlertSeverity
    alert_type: AlertType
    enabled: bool = True


@dataclass
class Alert:
    """Alert notification data"""

    symbol: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    current_price: float
    predicted_price: float
    price_change_pct: float
    timestamp: datetime
    metadata: Dict[str, Any]


class AlertingEngine:
    """Intelligent alerting system with configurable rules"""

    def __init__(self):
        self.alert_rules = self._initialize_default_rules()
        self.alert_history: List[Alert] = []

    def _initialize_default_rules(self) -> Dict[AlertType, AlertRule]:
        """Initialize sensible default alert rules"""
        return {
            AlertType.PRICE_INCREASE: AlertRule(
                threshold_pct=1.0,
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.PRICE_INCREASE,
            ),
            AlertType.PRICE_DECREASE: AlertRule(
                threshold_pct=-1.0,
                severity=AlertSeverity.MEDIUM,
                alert_type=AlertType.PRICE_DECREASE,
            ),
            AlertType.VOLATILITY_SPIKE: AlertRule(
                threshold_pct=3.0,
                severity=AlertSeverity.HIGH,
                alert_type=AlertType.VOLATILITY_SPIKE,
            ),
        }

    def evaluate_prediction(self, prediction_result: Dict[str, Any]) -> List[Alert]:
        """Evaluate prediction results against alert rules"""
        alerts = []

        symbol = prediction_result.get("symbol")
        current_price = prediction_result.get("current_price")
        predicted_price = prediction_result.get("predicted_price")
        price_change_pct = prediction_result.get("price_change_pct")

        if not all([symbol, current_price, predicted_price, price_change_pct]):
            logger.warning("Incomplete prediction result for alert evaluation")
            return alerts

        # Check price movement alerts
        alerts.extend(
            self._check_price_alerts(
                symbol, current_price, predicted_price, price_change_pct
            )
        )

        # Store alerts in history
        self.alert_history.extend(alerts)

        # Log alerts
        for alert in alerts:
            logger.warning(f"ALERT: {alert.message}")

        return alerts

    def _check_price_alerts(
        self,
        symbol: str,
        current_price: float,
        predicted_price: float,
        price_change_pct: float,
    ) -> List[Alert]:
        """Check for price-based alerts"""
        alerts = []

        # Price increase alert
        increase_rule = self.alert_rules.get(AlertType.PRICE_INCREASE)
        if (
            increase_rule
            and increase_rule.enabled
            and price_change_pct >= increase_rule.threshold_pct
        ):

            alert = Alert(
                symbol=symbol,
                alert_type=AlertType.PRICE_INCREASE,
                severity=increase_rule.severity,
                message=f"{symbol}: Predicted price increase of {price_change_pct:+.2f}% "
                f"(${current_price:.2f} → ${predicted_price:.2f})",
                current_price=current_price,
                predicted_price=predicted_price,
                price_change_pct=price_change_pct,
                timestamp=datetime.now(),
                metadata={"rule_threshold": increase_rule.threshold_pct},
            )
            alerts.append(alert)

        # Price decrease alert
        decrease_rule = self.alert_rules.get(AlertType.PRICE_DECREASE)
        if (
            decrease_rule
            and decrease_rule.enabled
            and price_change_pct <= decrease_rule.threshold_pct
        ):

            alert = Alert(
                symbol=symbol,
                alert_type=AlertType.PRICE_DECREASE,
                severity=decrease_rule.severity,
                message=f"{symbol}: Predicted price decrease of {price_change_pct:+.2f}% "
                f"(${current_price:.2f} → ${predicted_price:.2f})",
                current_price=current_price,
                predicted_price=predicted_price,
                price_change_pct=price_change_pct,
                timestamp=datetime.now(),
                metadata={"rule_threshold": decrease_rule.threshold_pct},
            )
            alerts.append(alert)

        return alerts

    def configure_alert_rule(self, alert_type: AlertType, rule: AlertRule) -> None:
        """Configure custom alert rule"""
        self.alert_rules[alert_type] = rule
        logger.info(f"Updated alert rule for {alert_type.value}: {rule.threshold_pct}%")

    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts sorted by timestamp"""
        return sorted(self.alert_history, key=lambda x: x.timestamp, reverse=True)[
            :limit
        ]
