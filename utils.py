"""
utils.py — Production Utilities (High-Bar V2)
Shared utilities for logging, math, config loading, and path management.
"""

import os
import json
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime


# ============================================================================
# CONFIG LOADER (V2)
# ============================================================================
class ConfigLoader:
    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """Loads YAML config with environment override support."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)

        # Apply environment-level overrides
        env = os.environ.get("ENV", "prod")
        if env in ("dev", "development"):
            if "system" not in cfg:
                cfg["system"] = {}
            cfg["system"]["log_level"] = "DEBUG"

        return cfg


# ============================================================================
# PRODUCTION LOGGER (V2)
# ============================================================================
class ProductionLogger:
    """
    High-Bar JSON Structured Logger
    - Per-run folder
    - Per-agent logs
    - JSON structured logs
    - decision_log.jsonl supported
    """

    def __init__(self, agent_name: str, base_log_dir: str = "logs", run_id: str = None, log_level: str = "INFO"):
        self.agent_name = agent_name.lower()
        self.run_id = run_id or f"run_{int(datetime.utcnow().timestamp())}"

        # Create run-level directory
        self.run_dir = os.path.join(base_log_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)

        # Log file path
        log_file = os.path.join(self.run_dir, f"{self.agent_name}.log")

        # Create Python logger
        self.logger = logging.getLogger(f"{self.agent_name}_{self.run_id}")

        # Avoid duplicate handlers when running tests repeatedly
        if self.logger.handlers:
            self.logger.handlers.clear()

        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

        # File handler
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Decision log (JSONL)
        self.decision_log_path = os.path.join(self.run_dir, "decision_log.jsonl")

    # --------------------- JSON Logging Helpers ---------------------
    def _log_json(self, level: str, message: str, **kwargs):
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "agent": self.agent_name,
            "run_id": self.run_id,
            "message": message,
            "details": kwargs if kwargs else {},
        }
        self.logger.info(json.dumps(payload))

    def info(self, msg: str, **kwargs): self._log_json("INFO", msg, **kwargs)
    def warning(self, msg: str, **kwargs): self._log_json("WARNING", msg, **kwargs)
    def error(self, msg: str, **kwargs): self._log_json("ERROR", msg, **kwargs)

    def exception(self, msg: str, **kwargs):
        kwargs["exception"] = True
        self._log_json("EXCEPTION", msg, **kwargs)

    # INPUT SUMMARY
    def log_input_summary(self, event: str, data: Dict[str, Any]):
        self.info("INPUT_SUMMARY", event=event, data=data)

    # OUTPUT SUMMARY
    def log_output_summary(self, event: str, data: Dict[str, Any]):
        self.info("OUTPUT_SUMMARY", event=event, data=data)

    # DECISION LOGGING (NEW — High-Bar Requirement)
    def log_decision(self, title: str, reasoning: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "title": title,
            "reasoning": reasoning,
            "details": details,
        }
        # Append JSONL
        with open(self.decision_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Also push to main log
        self.info("DECISION", title=title, reasoning=reasoning, details=details)


# ============================================================================
# MATH UTILITIES — High-Bar Safe Versions
# ============================================================================
def safe_divide(numerator: Union[float, int], denominator: Union[float, int], fill: float = 0.0) -> float:
    """Zero-division safe."""
    try:
        if denominator == 0 or denominator is None or pd.isna(denominator):
            return fill
        return float(numerator) / float(denominator)
    except Exception:
        return fill


def safe_percentage_change(baseline: float, current: float) -> Optional[float]:
    """
    Robust percentage change calculator.
    Returns None when baseline value is invalid or too small.
    """
    try:
        if baseline is None or baseline == 0 or pd.isna(baseline):
            return None
        if abs(baseline) < 1e-9:
            return None
        
        delta = float(current) - float(baseline)
        return (delta / float(baseline)) * 100.0
    except Exception:
        return None


def safe_value(v):
    """Converts numpy types → Python native for JSON safety."""
    if isinstance(v, (np.floating, np.float32, np.float64)):
        return float(v)
    if isinstance(v, (np.integer, np.int32, np.int64)):
        return int(v)
    return v
