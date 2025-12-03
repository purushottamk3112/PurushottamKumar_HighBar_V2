# """
# Evaluator V2 - High Bar Production System
# Performs: Welch T-Test, Mann-Whitney, Cliff's Delta, Drift Detection,
# Bootstrap CI, FDR, Confidence scoring, and Validation Logic.
# """

# import os
# import time
# import json
# from dataclasses import dataclass, asdict
# from typing import Dict, Any, List, Optional, Tuple

# import numpy as np
# import pandas as pd
# from scipy import stats
# from statsmodels.stats.power import TTestIndPower
# from statsmodels.stats.multitest import multipletests

# # Optional schema validation
# try:
#     from schema import DataSchema
# except ImportError:
#     DataSchema = None

# # Custom utilities
# from utils import ProductionLogger, safe_percentage_change


# # =====================================================================
# # Helpers for JSON-safety
# # =====================================================================

# def _json_safe(value: Any) -> Any:
#     """Convert numpy types to native Python for JSON serialization."""
#     if isinstance(value, (np.floating, np.float32, np.float64)):
#         return float(value)
#     if isinstance(value, (np.integer, np.int32, np.int64)):
#         return int(value)
#     if isinstance(value, (np.bool_, bool)):
#         return bool(value)
#     return value


# def _json_safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
#     return {k: _json_safe(v) for k, v in d.items()}


# # =====================================================================
# # Dataclasses
# # =====================================================================

# @dataclass
# class Evidence:
#     metric: str
#     baseline_value: float
#     current_value: float
#     absolute_delta: float
#     relative_delta_pct: float
#     sample_size_baseline: int
#     sample_size_current: int
#     p_value: float
#     adjusted_p_value: float
#     power: float
#     effect_size: float
#     bootstrap_ci: Dict[str, Any]
#     drift_detected: bool
#     segment: Optional[str] = None
#     segment_value: Optional[str] = None

#     def to_dict(self) -> Dict[str, Any]:
#         d = asdict(self)
#         # Make nested CI dict JSON-safe
#         if d["bootstrap_ci"]:
#             d["bootstrap_ci"] = _json_safe_dict(d["bootstrap_ci"])
#         # Top-level numeric / bool fields
#         d = _json_safe_dict(d)
#         return d


# @dataclass
# class ValidationResult:
#     hypothesis: str
#     validated: bool
#     impact: str
#     confidence: float
#     evidence: List[Evidence]
#     statistical_tests: Dict[str, Any]
#     severity: str
#     recommendation: str
#     run_id: Optional[str] = None
#     metadata: Dict[str, Any] = None

#     def to_dict(self) -> Dict[str, Any]:
#         d = asdict(self)
#         # Convert nested Evidence objects
#         d["evidence"] = [e.to_dict() for e in self.evidence]
#         # Make statistical_tests & metadata JSON-safe
#         d["statistical_tests"] = {
#             k: _json_safe_dict(v) if isinstance(v, dict) else v
#             for k, v in d.get("statistical_tests", {}).items()
#         }
#         if d.get("metadata"):
#             d["metadata"] = _json_safe_dict(d["metadata"])
#         # Top-level fields JSON-safe
#         d = _json_safe_dict(d)
#         return d


# # =====================================================================
# # Evaluator V2 (Production-Grade)
# # =====================================================================

# class Evaluator:
#     DEFAULT_CONFIG = {
#         "winsor_limits": (0.05, 0.05),
#         "bootstrap_iterations": 1000,
#         "bootstrap_ci_pct": 95,
#         "p_value_threshold": 0.05,
#         "confidence_minimum": 0.6,
#         "min_sample_size": 30,
#         "drift_pvalue_threshold": 0.05,
#         "confidence_weights": {
#             "significance": 0.4,
#             "power": 0.2,
#             "effect_size": 0.3,
#             "consistency": 0.1,
#         },
#         "fdr_alpha": 0.05,
#         "max_metrics_for_fdr": 200,
#     }

#     def __init__(self, config: Dict[str, Any], logger: ProductionLogger, run_id: str = None):
#         # Merge top-level config (the project uses full config.yaml here)
#         self.config = {**Evaluator.DEFAULT_CONFIG, **(config.get("evaluator", {}) if isinstance(config, dict) else {})}
#         self.logger = logger
#         self.thresholds = config.get("thresholds", {}) if isinstance(config, dict) else {}
#         self.schema_config = config.get("schema", {"strict_mode": True}) if isinstance(config, dict) else {"strict_mode": True}
#         self.run_id = run_id or f"run_{int(time.time())}"
#         self.weights = self.config.get("confidence_weights", Evaluator.DEFAULT_CONFIG["confidence_weights"])

#         self.logger.info(f"Evaluator initialized (run_id={self.run_id})")

#         if hasattr(self.logger, "run_dir") and self.logger.run_dir:
#             os.makedirs(self.logger.run_dir, exist_ok=True)

#     # =====================================================================
#     # Entry point for Orchestrator
#     # =====================================================================

#     def evaluate_insight(self, insight, baseline_df, current_df):
#         """
#         Orchestrator passes full Insight object.
#         """
#         hypothesis = insight.hypothesis
#         metrics = insight.affected_metrics
#         segments = insight.affected_segments
#         expected_direction = insight.expected_direction

#         return self.evaluate_hypothesis(
#             hypothesis=hypothesis,
#             baseline_data=baseline_df,
#             current_data=current_df,
#             metrics=metrics,
#             segments=segments,
#             expected_direction=expected_direction,
#         )

#     # =====================================================================
#     # Evaluation Pipeline
#     # =====================================================================

#     def evaluate_hypothesis(
#         self,
#         hypothesis: str,
#         baseline_data: pd.DataFrame,
#         current_data: pd.DataFrame,
#         metrics: List[str],
#         segments: Optional[Dict[str, str]] = None,
#         expected_direction: str = "increase",
#     ) -> ValidationResult:

#         self.logger.info(f"Starting evaluation: {hypothesis}")

#         results_map: Dict[str, Dict[str, Any]] = {}
#         raw_p_values: List[float] = []
#         valid_metrics: List[str] = []
#         statistical_tests: Dict[str, Any] = {}

#         # Validate inputs
#         try:
#             self._validate_inputs(baseline_data, current_data, metrics)
#         except Exception as e:
#             self.logger.exception(f"Pre-run validation failed: {e}")
#             return self._fallback_result(hypothesis, f"Pre-run validation failed: {e}")

#         # Apply segment filtering
#         baseline_filtered = self._apply_filters(baseline_data, segments)
#         current_filtered = self._apply_filters(current_data, segments)

#         # Analyze each metric
#         for metric in metrics:
#             try:
#                 base_series = baseline_filtered[metric].replace([np.inf, -np.inf], np.nan).dropna()
#                 curr_series = current_filtered[metric].replace([np.inf, -np.inf], np.nan).dropna()

#                 # Sample size guardrail
#                 if len(base_series) < self.config["min_sample_size"] or len(curr_series) < self.config["min_sample_size"]:
#                     self.logger.warning(f"Insufficient data for metric '{metric}'")
#                     continue

#                 # Robust winsorization
#                 base_w = self._winsorize(base_series, tuple(self.config["winsor_limits"]))
#                 curr_w = self._winsorize(curr_series, tuple(self.config["winsor_limits"]))

#                 res = self._analyze_metric(metric, base_w, curr_w)
#                 if "error" in res:
#                     self.logger.warning(f"Metric '{metric}' analysis error: {res['error']}")
#                     continue

#                 results_map[metric] = res

#                 if "p_value" in res:
#                     raw_p_values.append(float(res["p_value"]))
#                     valid_metrics.append(metric)

#                 statistical_tests[metric] = {
#                     "t_stat": _json_safe(res.get("t_statistic")),
#                     "p_value": _json_safe(res.get("p_value")),
#                     "ks_stat": _json_safe(res.get("ks_statistic")),
#                     "drift_detected": _json_safe(res.get("drift_detected")),
#                     "effect_size": _json_safe(res.get("effect_size")),
#                     "power": _json_safe(res.get("power")),
#                 }

#             except Exception as e:
#                 self.logger.exception(f"Metric analysis failed for '{metric}': {e}")
#                 continue

#         # FDR correction
#         adjusted_p_map: Dict[str, float] = {}
#         if raw_p_values and len(raw_p_values) <= self.config["max_metrics_for_fdr"]:
#             try:
#                 adjusted_list = self._apply_fdr_raw(raw_p_values)
#                 adjusted_p_map = dict(zip(valid_metrics, adjusted_list))
#             except Exception as e:
#                 self.logger.error(f"FDR correction failed: {e}")
#                 adjusted_p_map = dict(zip(valid_metrics, raw_p_values))
#         else:
#             adjusted_p_map = dict(zip(valid_metrics, raw_p_values))

#         # Build evidence list
#         evidence_list: List[Evidence] = []
#         for metric in valid_metrics:
#             res = results_map.get(metric, {})
#             if not res:
#                 continue

#             adj_p = float(adjusted_p_map.get(metric, res.get("p_value", 1.0)))
#             ci = self._bootstrap_ci(
#                 res.get("base_vals_arr"),
#                 res.get("curr_vals_arr"),
#                 self.config["bootstrap_iterations"],
#             )

#             rel_delta = res.get("relative_delta_pct", 0.0)
#             if rel_delta is None:
#                 rel_delta = 0.0

#             evidence_list.append(
#                 Evidence(
#                     metric=metric,
#                     baseline_value=float(res["baseline_mean"]),
#                     current_value=float(res["current_mean"]),
#                     absolute_delta=float(res["absolute_delta"]),
#                     relative_delta_pct=float(rel_delta),
#                     sample_size_baseline=int(res["n_base"]),
#                     sample_size_current=int(res["n_curr"]),
#                     p_value=float(res["p_value"]),
#                     adjusted_p_value=adj_p,
#                     power=float(res.get("power", 0.0)),
#                     effect_size=float(res.get("effect_size", 0.0)),
#                     bootstrap_ci=ci,
#                     drift_detected=bool(res.get("drift_detected", False)),
#                     segment=list(segments.keys())[0] if segments else None,
#                     segment_value=list(segments.values())[0] if segments else None,
#                 )
#             )

#         # Final signals
#         impact = self._assess_impact(evidence_list)
#         severity = self._assess_severity(evidence_list)
#         confidence = self._compute_confidence(evidence_list)
#         validated = self._is_validated(evidence_list, confidence, expected_direction)
#         recommendation = self._generate_recommendation(hypothesis, evidence_list, impact, severity, validated)

#         result = ValidationResult(
#             hypothesis=hypothesis,
#             validated=validated,
#             impact=impact,
#             confidence=float(round(confidence, 4)),
#             evidence=evidence_list,
#             statistical_tests=statistical_tests,
#             severity=severity,
#             recommendation=recommendation,
#             run_id=self.run_id,
#             metadata={
#                 "method": "Winsorized Welch + Mann-Whitney + Cliff's Delta",
#                 "bootstrap_iterations": int(self.config["bootstrap_iterations"]),
#                 "timestamp": pd.Timestamp.utcnow().isoformat(),
#             },
#         )

#         self._write_decision_log(result)
#         self.logger.log_output_summary(
#             "EVALUATION_SUMMARY",
#             {
#                 "hypothesis": hypothesis,
#                 "validated": validated,
#                 "confidence": float(result.confidence),
#                 "run_id": self.run_id,
#             },
#         )

#         return result

#     # =====================================================================
#     # Metric-Level Analysis
#     # =====================================================================

#     def _analyze_metric(self, metric: str, base: pd.Series, curr: pd.Series) -> Dict[str, Any]:
#         try:
#             base_arr = np.asarray(base, dtype=float)
#             curr_arr = np.asarray(curr, dtype=float)

#             n_base, n_curr = base_arr.size, curr_arr.size
#             if n_base < 2 or n_curr < 2:
#                 return {"error": "Insufficient sample size (<2)"}

#             mean_base = float(np.mean(base_arr))
#             mean_curr = float(np.mean(curr_arr))

#             # Welch t-test
#             t_stat, p_val = stats.ttest_ind(base_arr, curr_arr, equal_var=False)

#             # Mann-Whitney
#             try:
#                 u_stat, mw_p = stats.mannwhitneyu(base_arr, curr_arr, alternative="two-sided")
#             except Exception:
#                 u_stat, mw_p = 0.0, 1.0

#             cliffs_delta = (2.0 * u_stat - (n_base * n_curr)) / (n_base * n_curr)

#             # KS drift detection
#             try:
#                 ks_stat, ks_p = stats.ks_2samp(base_arr, curr_arr)
#                 drift = bool(ks_p < self.config["drift_pvalue_threshold"])
#             except Exception:
#                 ks_stat, drift = 0.0, False

#             # Power (Cohen's d → TTestIndPower)
#             power = 0.0
#             try:
#                 pooled_std = np.sqrt(
#                     ((n_base - 1) * np.var(base_arr, ddof=1) +
#                      (n_curr - 1) * np.var(curr_arr, ddof=1)) /
#                     (n_base + n_curr - 2)
#                 )
#                 if pooled_std > 0:
#                     cohens_d = abs(mean_curr - mean_base) / pooled_std
#                     analysis = TTestIndPower()
#                     power = float(
#                         analysis.solve_power(
#                             effect_size=cohens_d,
#                             nobs1=n_curr,
#                             ratio=n_curr / n_base,
#                             alpha=0.05,
#                         )
#                     )
#             except Exception:
#                 power = 0.0

#             rel_delta = safe_percentage_change(mean_base, mean_curr)
#             if rel_delta is None:
#                 rel_delta = 0.0

#             return {
#                 "baseline_mean": mean_base,
#                 "current_mean": mean_curr,
#                 "absolute_delta": mean_curr - mean_base,
#                 "relative_delta_pct": rel_delta,
#                 "n_base": n_base,
#                 "n_curr": n_curr,
#                 "p_value": float(p_val),
#                 "t_statistic": float(t_stat),
#                 "ks_statistic": float(ks_stat),
#                 "drift_detected": drift,
#                 "effect_size": float(cliffs_delta),
#                 "power": power,
#                 "base_vals_arr": base_arr,
#                 "curr_vals_arr": curr_arr,
#             }

#         except Exception as e:
#             return {"error": str(e)}

#     # =====================================================================
#     # Helper Utilities
#     # =====================================================================

#     def _apply_filters(self, df: pd.DataFrame, segments: Optional[Dict[str, str]]) -> pd.DataFrame:
#         if not segments:
#             return df
#         filtered = df.copy()
#         for col, val in segments.items():
#             if col in filtered.columns:
#                 filtered = filtered[filtered[col] == val]
#         return filtered

#     def _validate_inputs(self, baseline: pd.DataFrame, current: pd.DataFrame, metrics: List[str]):
#         if baseline is None or current is None:
#             raise ValueError("Baseline or current dataframe is None")

#         if DataSchema:
#             schema = DataSchema()
#             vb = schema.validate_dataframe(baseline)
#             if not vb.get("valid", True):
#                 raise ValueError(f"Baseline schema errors: {vb.get('errors')}")
#             vc = schema.validate_dataframe(current)
#             if not vc.get("valid", True):
#                 raise ValueError(f"Current schema errors: {vc.get('errors')}")

#         for m in metrics:
#             if m not in baseline.columns or m not in current.columns:
#                 raise ValueError(f"Missing metric column '{m}'")

#     def _winsorize(self, series: pd.Series, limits: Tuple[float, float]) -> pd.Series:
#         try:
#             arr = np.asarray(series, dtype=float)
#             return pd.Series(stats.mstats.winsorize(arr, limits=limits))
#         except Exception:
#             return series

#     def _bootstrap_ci(self, base: np.ndarray, curr: np.ndarray, iterations: int) -> Dict[str, Any]:
#         if base is None or curr is None:
#             return {}
#         try:
#             rng = np.random.default_rng()
#             diffs = [
#                 float(np.mean(rng.choice(curr, size=len(curr), replace=True)) -
#                       np.mean(rng.choice(base, size=len(base), replace=True)))
#                 for _ in range(iterations)
#             ]
#             lower = float(np.percentile(diffs, 2.5))
#             upper = float(np.percentile(diffs, 97.5))
#             return {
#                 "ci_lower_95": lower,
#                 "ci_upper_95": upper,
#                 "zero_included": bool(lower <= 0 <= upper),
#             }
#         except Exception:
#             return {"ci_lower_95": None, "ci_upper_95": None, "zero_included": True}

#     def _apply_fdr_raw(self, p_values: List[float]) -> List[float]:
#         _, adj, _, _ = multipletests(
#             p_values,
#             alpha=self.config["fdr_alpha"],
#             method="fdr_bh",
#         )
#         return [float(a) for a in adj]

#     # =====================================================================
#     # Confidence & Interpretation
#     # =====================================================================

#     def _compute_confidence(self, evidence: List[Evidence]) -> float:
#         if not evidence:
#             return 0.0

#         scores = []
#         for ev in evidence:
#             s = 0.0
#             if ev.adjusted_p_value < self.config["p_value_threshold"]:
#                 s += self.weights["significance"]
#             if ev.power > 0.8:
#                 s += self.weights["power"]
#             if abs(ev.effect_size) > 0.147:  # small effect size threshold
#                 s += self.weights["effect_size"]
#             if not ev.drift_detected:
#                 s += self.weights["consistency"]
#             scores.append(s)

#         return min(round(float(np.mean(scores)), 2), 1.0)

#     def _is_validated(self, evidence: List[Evidence], confidence: float, expected_direction: str) -> bool:
#         if confidence < self.config["confidence_minimum"]:
#             return False

#         for ev in evidence:
#             if expected_direction == "increase" and ev.relative_delta_pct > 0:
#                 return True
#             if expected_direction == "decrease" and ev.relative_delta_pct < 0:
#                 return True

#         return False

#     def _assess_impact(self, evidence: List[Evidence]) -> str:
#         roas_severe = float(self.thresholds.get("roas_drop_severe", 0.30)) * 100
#         ctr_severe = float(self.thresholds.get("ctr_drop_severe", 0.25)) * 100

#         for ev in evidence:
#             if ev.metric.lower() == "roas" and ev.relative_delta_pct < -roas_severe:
#                 return "high"
#             if ev.metric.lower() == "ctr" and ev.relative_delta_pct < -ctr_severe:
#                 return "high"

#         # Could expand with medium / low bands; for now "low" means non-severe
#         return "low"

#     def _assess_severity(self, evidence: List[Evidence]) -> str:
#         roas_severe = float(self.thresholds.get("roas_drop_severe", 0.30)) * 100
#         roas_moderate = float(self.thresholds.get("roas_drop_moderate", 0.15)) * 100

#         drops = [ev.relative_delta_pct for ev in evidence if ev.relative_delta_pct < 0]
#         max_drop = min(drops) if drops else 0

#         if max_drop <= -roas_severe:
#             return "severe"
#         if max_drop <= -roas_moderate:
#             return "moderate"
#         return "minor"

#     def _generate_recommendation(
#         self,
#         hypothesis: str,
#         evidence: List[Evidence],
#         impact: str,
#         severity: str,
#         validated: bool,
#     ) -> str:
#         if not validated:
#             return "Hypothesis not statistically validated."
#         return f"Validated hypothesis. Impact={impact}, Severity={severity}. Recommend action based on trend."

#     def _fallback_result(self, hypothesis: str, error: str) -> ValidationResult:
#         return ValidationResult(
#             hypothesis=hypothesis,
#             validated=False,
#             impact="unknown",
#             confidence=0.0,
#             evidence=[],
#             statistical_tests={},
#             severity="unknown",
#             recommendation=f"Error: {error}",
#             run_id=self.run_id,
#             metadata={},
#         )

#     # =====================================================================
#     # Logging
#     # =====================================================================

#     def _write_decision_log(self, result: ValidationResult):
#         if not self.logger:
#             return
#         try:
#             path = os.path.join(self.logger.run_dir, "decision_log.jsonl")
#             with open(path, "a") as f:
#                 f.write(json.dumps(result.to_dict()) + "\n")
#         except Exception:
#             # Logging must never crash the pipeline
#             pass


# if __name__ == "__main__":
#     print("Evaluator V2 High-Bar loaded.")















"""
Evaluator V2 - High Bar Production System
Performs: Welch T-Test, Mann-Whitney, Cliff's Delta, Drift Detection,
Bootstrap CI, FDR, Confidence scoring, and Validation Logic.
"""

import os
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multitest import multipletests

# Optional schema validation
try:
    from schema import DataSchema
except ImportError:
    DataSchema = None

# Custom utilities
from utils import ProductionLogger, safe_percentage_change, safe_divide


# =====================================================================
# Helpers for JSON-safety
# =====================================================================

def _json_safe(value: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _json_safe_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: _json_safe(v) for k, v in d.items()}


# =====================================================================
# Dataclasses
# =====================================================================

@dataclass
class Evidence:
    metric: str
    baseline_value: float
    current_value: float
    absolute_delta: float
    relative_delta_pct: float
    sample_size_baseline: int
    sample_size_current: int
    p_value: float
    adjusted_p_value: float
    power: float
    effect_size: float
    bootstrap_ci: Dict[str, Any]
    drift_detected: bool
    segment: Optional[str] = None
    segment_value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Ensure recursive safety
        return _json_safe_dict(d)


@dataclass
class ValidationResult:
    hypothesis: str
    validated: bool
    impact: str
    confidence: float
    evidence: List[Evidence]
    statistical_tests: Dict[str, Any]
    severity: str
    recommendation: str
    run_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Convert nested Evidence objects
        d["evidence"] = [e.to_dict() for e in self.evidence]
        # Make everything else JSON safe
        return _json_safe_dict(d)


# =====================================================================
# Evaluator V2 (Production-Grade)
# =====================================================================

class Evaluator:
    DEFAULT_CONFIG = {
        "winsor_limits": (0.05, 0.05),
        "bootstrap_iterations": 1000,
        "bootstrap_ci_pct": 95,
        "p_value_threshold": 0.05,
        "confidence_minimum": 0.6,
        "min_sample_size": 30,
        "drift_pvalue_threshold": 0.05,
        "confidence_weights": {
            "significance": 0.4,
            "power": 0.2,
            "effect_size": 0.3,
            "consistency": 0.1,
        },
        "fdr_alpha": 0.05,
        "max_metrics_for_fdr": 200,
    }

    DERIVED_METRIC_FUNCS = {
        "roas": lambda df: safe_divide(df.get("revenue", pd.Series(dtype=float)), df.get("spend", pd.Series(dtype=float))),
        "ctr": lambda df: safe_divide(df.get("clicks", pd.Series(dtype=float)), df.get("impressions", pd.Series(dtype=float))),
        "cpa": lambda df: safe_divide(df.get("spend", pd.Series(dtype=float)), df.get("purchases", pd.Series(dtype=float))),
        "cvr": lambda df: safe_divide(df.get("purchases", pd.Series(dtype=float)), df.get("clicks", pd.Series(dtype=float))),
    }

    def __init__(self, config: Dict[str, Any], logger: ProductionLogger, run_id: str = None):
        # Merge top-level config (the project uses full config.yaml here)
        self.config = {**Evaluator.DEFAULT_CONFIG, **(config.get("evaluator", {}) if isinstance(config, dict) else {})}
        self.logger = logger
        self.thresholds = config.get("thresholds", {}) if isinstance(config, dict) else {}
        self.schema_config = config.get("schema", {"strict_mode": True}) if isinstance(config, dict) else {"strict_mode": True}
        self.run_id = run_id or f"run_{int(time.time())}"
        self.weights = self.config.get("confidence_weights", Evaluator.DEFAULT_CONFIG["confidence_weights"])

        self.logger.info(f"Evaluator initialized (run_id={self.run_id})")

        if hasattr(self.logger, "run_dir") and self.logger.run_dir:
            os.makedirs(self.logger.run_dir, exist_ok=True)

    # =====================================================================
    # Entry point for Orchestrator
    # =====================================================================

    def evaluate_insight(self, insight, baseline_df, current_df):
        """
        Orchestrator passes full Insight object.
        """
        hypothesis = getattr(insight, 'hypothesis', 'unknown')
        metrics = getattr(insight, 'affected_metrics', [])
        segments = getattr(insight, 'affected_segments', {})
        expected_direction = getattr(insight, 'expected_direction', 'decrease')
        confidence_hint = float(getattr(insight, 'confidence_hint', 0.0))

        return self.evaluate_hypothesis(
            hypothesis=hypothesis,
            baseline_data=baseline_df,
            current_data=current_df,
            metrics=metrics,
            segments=segments,
            expected_direction=expected_direction,
            confidence_hint=confidence_hint
        )

    # =====================================================================
    # Evaluation Pipeline
    # =====================================================================

    def evaluate_hypothesis(
        self,
        hypothesis: str,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        metrics: List[str],
        segments: Optional[Dict[str, str]] = None,
        expected_direction: str = "increase",
        confidence_hint: float = 0.0
    ) -> ValidationResult:

        self.logger.info(f"Starting evaluation: {hypothesis}")

        results_map = {}
        raw_p_values = []
        valid_metrics = []
        statistical_tests = {}

        # Validate inputs
        try:
            self._validate_inputs(baseline_data, current_data, metrics)
        except Exception as e:
            self.logger.exception(f"Pre-run validation failed: {e}")
            return self._fallback_result(hypothesis, str(e))

        # Apply segment filtering
        baseline_filtered = self._apply_filters(baseline_data, segments)
        current_filtered = self._apply_filters(current_data, segments)

        # Analyze each metric
        for metric in metrics:
            try:
                # Build metric series (derived or raw)
                base_series, curr_series = self._build_metric_series(metric, baseline_filtered, current_filtered)
                
                # Clean
                base_series = base_series.replace([np.inf, -np.inf], np.nan).dropna()
                curr_series = curr_series.replace([np.inf, -np.inf], np.nan).dropna()

                min_n = int(self.config.get("min_sample_size", 5))
                if len(base_series) < min_n or len(curr_series) < min_n:
                    self.logger.warning(f"Insufficient data for metric '{metric}'")
                    continue

                # Winsorize
                base_w = self._winsorize(base_series, tuple(self.config["winsor_limits"]))
                curr_w = self._winsorize(curr_series, tuple(self.config["winsor_limits"]))

                res = self._analyze_metric(metric, base_w, curr_w)
                results_map[metric] = res

                if "p_value" in res:
                    raw_p_values.append(res["p_value"])
                    valid_metrics.append(metric)

                statistical_tests[metric] = {
                    "t_stat": res.get("t_statistic"),
                    "p_value": res.get("p_value"),
                    "drift_detected": res.get("drift_detected"),
                    "effect_size": res.get("effect_size"),
                    "power": res.get("power")
                }

            except Exception as e:
                self.logger.exception(f"Metric analysis failed for '{metric}': {e}")
                results_map[metric] = {"error": str(e)}
                continue

        # FDR correction
        adjusted_p_map = {}
        if raw_p_values:
            try:
                if len(raw_p_values) <= self.config["max_metrics_for_fdr"]:
                    adjusted_list = self._apply_fdr_raw(raw_p_values)
                    adjusted_p_map = dict(zip(valid_metrics, adjusted_list))
                else:
                    adjusted_p_map = dict(zip(valid_metrics, raw_p_values))
            except:
                adjusted_p_map = dict(zip(valid_metrics, raw_p_values))

        # Build evidence list
        evidence_list = []
        for metric in valid_metrics:
            res = results_map.get(metric, {})
            if "error" in res: continue

            adj_p = float(adjusted_p_map.get(metric, res.get("p_value", 1.0)))
            ci = self._bootstrap_ci(
                res.get("base_vals_arr"),
                res.get("curr_vals_arr"),
                self.config["bootstrap_iterations"]
            )

            evidence_list.append(
                Evidence(
                    metric=metric,
                    baseline_value=float(res["baseline_mean"]),
                    current_value=float(res["current_mean"]),
                    absolute_delta=float(res["absolute_delta"]),
                    relative_delta_pct=float(res["relative_delta_pct"]),
                    sample_size_baseline=int(res["n_base"]),
                    sample_size_current=int(res["n_curr"]),
                    p_value=float(res["p_value"]),
                    adjusted_p_value=adj_p,
                    power=float(res["power"]),
                    effect_size=float(res["effect_size"]),
                    bootstrap_ci=ci,
                    drift_detected=bool(res["drift_detected"]),
                    segment=list(segments.keys())[0] if segments else None,
                    segment_value=list(segments.values())[0] if segments else None
                )
            )

        # Final signals
        impact = self._assess_impact(evidence_list)
        severity = self._assess_severity(evidence_list)
        confidence = self._compute_confidence(evidence_list, confidence_hint)
        validated = self._is_validated(evidence_list, confidence, expected_direction)
        recommendation = self._generate_recommendation(hypothesis, evidence_list, impact, severity, validated)

        result = ValidationResult(
            hypothesis=hypothesis,
            validated=validated,
            impact=impact,
            confidence=confidence,
            evidence=evidence_list,
            statistical_tests=statistical_tests,
            severity=severity,
            recommendation=recommendation,
            run_id=self.run_id,
            metadata={
                "method": "Winsorized Welch + Mann-Whitney + Cliff's Delta",
                "timestamp": pd.Timestamp.utcnow().isoformat()
            }
        )

        self._write_decision_log(result)
        self.logger.log_output_summary(
            "EVALUATION_SUMMARY",
            {
                "hypothesis": hypothesis,
                "validated": validated,
                "confidence": confidence,
                "run_id": self.run_id
            }
        )

        return result

    # =====================================================================
    # Metric-Level Analysis
    # =====================================================================
    
    def _build_metric_series(self, metric, base_df, curr_df):
        m = metric.lower()
        if m in self.DERIVED_METRIC_FUNCS:
            # FIX: Force conversion to Series so .replace() always works
            return (
                pd.Series(self.DERIVED_METRIC_FUNCS[m](base_df)),
                pd.Series(self.DERIVED_METRIC_FUNCS[m](curr_df))
            )
        return (
            base_df.get(metric, pd.Series(dtype=float)),
            curr_df.get(metric, pd.Series(dtype=float))
        )

    def _analyze_metric(self, metric, base, curr):
        try:
            base_arr = np.asarray(base, dtype=float)
            curr_arr = np.asarray(curr, dtype=float)

            n_base, n_curr = base_arr.size, curr_arr.size

            mean_base = float(np.mean(base_arr))
            mean_curr = float(np.mean(curr_arr))

            # Welch t-test
            t_stat, p_val = stats.ttest_ind(base_arr, curr_arr, equal_var=False)

            # Mann-Whitney
            try:
                u_stat, mw_p = stats.mannwhitneyu(base_arr, curr_arr, alternative="two-sided")
            except:
                u_stat, mw_p = 0.0, 1.0

            cliffs_delta = (2.0 * u_stat - (n_base * n_curr)) / (n_base * n_curr) if (n_base * n_curr) > 0 else 0.0

            # KS drift
            try:
                ks_stat, ks_p = stats.ks_2samp(base_arr, curr_arr)
                drift = ks_p < self.config["drift_pvalue_threshold"]
            except:
                ks_stat, drift = 0.0, False

            # Power (Cohen's d → TTestIndPower)
            power = 0.0
            try:
                pooled_std = np.sqrt(
                    ((n_base - 1) * np.var(base_arr, ddof=1) +
                     (n_curr - 1) * np.var(curr_arr, ddof=1)) /
                    (n_base + n_curr - 2)
                )
                if pooled_std > 0:
                    cohens_d = abs(mean_curr - mean_base) / pooled_std
                    analysis = TTestIndPower()
                    power = float(
                        analysis.solve_power(
                            effect_size=cohens_d,
                            nobs1=n_curr,
                            ratio=n_curr / n_base,
                            alpha=0.05
                        )
                    )
            except:
                pass

            rel_delta = safe_percentage_change(mean_base, mean_curr)
            if rel_delta is None:
                rel_delta = 0.0

            return {
                "baseline_mean": mean_base,
                "current_mean": mean_curr,
                "absolute_delta": mean_curr - mean_base,
                "relative_delta_pct": rel_delta,
                "n_base": n_base,
                "n_curr": n_curr,
                "p_value": float(p_val),
                "t_statistic": float(t_stat),
                "ks_statistic": float(ks_stat),
                "drift_detected": drift,
                "effect_size": float(cliffs_delta),
                "power": power,
                "base_vals_arr": base_arr,
                "curr_vals_arr": curr_arr
            }

        except Exception as e:
            return {"error": str(e)}

    # =====================================================================
    # Helper Utilities
    # =====================================================================

    def _apply_filters(self, df, segments):
        if not segments:
            return df
        filtered = df.copy()
        for col, val in segments.items():
            if col in filtered.columns:
                filtered = filtered[filtered[col] == val]
        return filtered

    def _validate_inputs(self, baseline, current, metrics):
        if baseline is None or current is None:
            raise ValueError("Baseline or current dataframe is None")

        # Allow metrics if they are derived (in DERIVED_METRIC_FUNCS) even if not in columns
        for m in metrics:
            if m not in baseline.columns and m.lower() not in self.DERIVED_METRIC_FUNCS:
                raise ValueError(f"Missing metric column '{m}'")

    def _winsorize(self, series, limits):
        try:
            arr = np.asarray(series, dtype=float)
            return pd.Series(stats.mstats.winsorize(arr, limits=limits))
        except:
            return series

    def _bootstrap_ci(self, base, curr, iterations):
        if base is None or curr is None:
            return {}
        try:
            rng = np.random.default_rng()
            diffs = [
                np.mean(rng.choice(curr, size=len(curr), replace=True)) -
                np.mean(rng.choice(base, size=len(base), replace=True))
                for _ in range(iterations)
            ]
            lower = float(np.percentile(diffs, 2.5))
            upper = float(np.percentile(diffs, 97.5))
            return {
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "zero_included": (lower <= 0 <= upper)
            }
        except:
            return {"ci_lower_95": None, "ci_upper_95": None, "zero_included": True}

    def _apply_fdr_raw(self, p_values):
        _, adj, _, _ = multipletests(
            p_values,
            alpha=self.config["fdr_alpha"],
            method="fdr_bh"
        )
        return list(adj)

    # =====================================================================
    # Confidence & Interpretation
    # =====================================================================

    def _compute_confidence(self, evidence, insight_hint):
        if not evidence:
            # If no evidence generated (e.g. all filtered out), rely on hint but capped low
            return min(insight_hint, 0.3)

        scores = []
        for ev in evidence:
            s = 0.0
            if ev.adjusted_p_value < self.config["p_value_threshold"]:
                s += self.weights["significance"]
            if ev.power > 0.8:
                s += self.weights["power"]
            if abs(ev.effect_size) > 0.147:
                s += self.weights["effect_size"]
            if not ev.drift_detected:
                s += self.weights["consistency"]
            scores.append(s)

        evidence_score = float(np.mean(scores))
        
        # Blend: 70% Evidence, 30% Prior Hint
        final_score = 0.7 * evidence_score + 0.3 * insight_hint
        return min(max(round(final_score, 4), 0.0), 1.0)

    def _is_validated(self, evidence, conf, direction):
        if conf < self.config["confidence_minimum"]:
            return False

        # At least one metric must move in the expected direction
        for ev in evidence:
            if direction == "increase" and ev.relative_delta_pct > 0:
                return True
            if direction == "decrease" and ev.relative_delta_pct < 0:
                return True
        return False

    def _assess_impact(self, evidence):
        roas_severe = float(self.thresholds.get("roas_drop_severe", 0.30)) * 100
        ctr_severe = float(self.thresholds.get("ctr_drop_severe", 0.25)) * 100

        for ev in evidence:
            if ev.metric.lower() == "roas" and ev.relative_delta_pct < -roas_severe:
                return "high"
            if ev.metric.lower() == "ctr" and ev.relative_delta_pct < -ctr_severe:
                return "high"
        return "low"

    def _assess_severity(self, evidence):
        roas_severe = float(self.thresholds.get("roas_drop_severe", 0.30)) * 100
        drops = [ev.relative_delta_pct for ev in evidence if ev.relative_delta_pct < 0]
        max_drop = min(drops) if drops else 0

        if max_drop <= -roas_severe:
            return "severe"
        return "moderate"

    def _generate_recommendation(self, hypothesis, evidence, impact, severity, validated):
        if not validated:
            return "Hypothesis not statistically validated."
        return f"Validated. Impact={impact}, Severity={severity}."

    def _fallback_result(self, hypothesis, error):
        return ValidationResult(
            hypothesis=hypothesis,
            validated=False,
            impact="unknown",
            confidence=0.0,
            evidence=[],
            statistical_tests={},
            severity="unknown",
            recommendation=f"Error: {error}",
            run_id=self.run_id,
            metadata={}
        )

    # =====================================================================
    # Logging
    # =====================================================================

    def _write_decision_log(self, result):
        if not self.logger:
            return
        try:
            # Use _json_safe_dict via to_dict to prevent bool_ errors
            path = os.path.join(self.logger.run_dir, "decision_log.jsonl")
            with open(path, "a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    print("Evaluator V2 High-Bar loaded.")