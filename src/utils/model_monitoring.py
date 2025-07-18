# src/utils/model_monitoring.py
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from loguru import logger


class ModelMonitor:
    """Monitor model performance and provide insights"""

    def __init__(self):
        self.performance_history = []

    def evaluate_prediction_quality(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if model performance is acceptable"""
        quality_assessment = {
            "overall_quality": "unknown",
            "recommendations": [],
            "alerts": [],
        }

        r2_score = metrics.get("r2_score", 0)
        if r2_score > 0.7:
            quality_assessment["overall_quality"] = "good"
        elif r2_score > 0.5:
            quality_assessment["overall_quality"] = "acceptable"
        else:
            quality_assessment["overall_quality"] = "poor"
            quality_assessment["alerts"].append(f"Low R² score: {r2_score:.3f}")

        directional_accuracy = metrics.get("directional_accuracy", 0.5)
        if directional_accuracy < 0.55:
            quality_assessment["recommendations"].append(
                "Consider additional features to improve directional accuracy"
            )

        train_r2 = metrics.get("train_r2", 0)
        if train_r2 - r2_score > 0.2:
            quality_assessment["alerts"].append("Potential overfitting detected")

        return quality_assessment

    def generate_performance_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable performance report"""
        report = f"""
📊 Model Performance Report
==========================

📈 Accuracy Metrics:
- R² Score: {metrics.get('r2_score', 0):.4f}
- Mean Squared Error: {metrics.get('mse', 0):.4f}
- Root Mean Squared Error: {metrics.get('rmse', 0):.4f}
- Mean Absolute Error: {metrics.get('mae', 0):.4f}

🎯 Directional Accuracy: {metrics.get('directional_accuracy', 0.5):.2%}

📊 Training Data:
- Training Samples: {metrics.get('n_train_samples', 0)}
- Test Samples: {metrics.get('n_test_samples', 0)}
- Features Used: {metrics.get('n_features', 0)}

🔍 Quality Assessment:
"""

        quality = self.evaluate_prediction_quality(metrics)
        report += f"- Overall Quality: {quality['overall_quality'].upper()}\n"

        if quality["alerts"]:
            report += "⚠️ Alerts:\n"
            for alert in quality["alerts"]:
                report += f"  - {alert}\n"

        if quality["recommendations"]:
            report += "💡 Recommendations:\n"
            for rec in quality["recommendations"]:
                report += f"  - {rec}\n"

        return report
