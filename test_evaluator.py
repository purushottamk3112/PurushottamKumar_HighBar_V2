"""
Unit Tests for Evaluator V2 (Production Grade)
Covers:
- End-to-end: Insight → Evaluator → ValidationResult
- Significant metric drops
- No-change scenarios
- Segment filtering
- NaN / Inf robustness
- Evidence & result structure
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from evaluator import Evaluator, Evidence, ValidationResult
from insight_agent import Insight, InsightEvidence
from utils import ProductionLogger


class TestEvaluatorV2(unittest.TestCase):
    """End-to-end tests for Evaluator V2 using synthetic data."""

    @classmethod
    def setUpClass(cls):
        # Minimal, evaluator-focused config
        cls.config = {
            "evaluator": {
                "min_sample_size": 5,
                "winsor_limits": (0.0, 0.0),
                "bootstrap_iterations": 200,
                "p_value_threshold": 0.05,
                "confidence_minimum": 0.6,
                "max_metrics_for_fdr": 50,
            },
            "thresholds": {
                "roas_drop_severe": 0.30,   # 30% drop
                "roas_drop_moderate": 0.15, # 15% drop
            },
            "confidence_weights": {
                "significance": 0.4,
                "power": 0.2,
                "effect_size": 0.3,
                "consistency": 0.1,
            },
        }

        cls.logger = ProductionLogger("TestEvaluatorV2", Path("logs"))
        cls.evaluator = Evaluator(cls.config, cls.logger, run_id="test_run_evaluator_v2")

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def _make_base_current_df_roas_drop(self, n: int = 100):
        """
        Create synthetic baseline/current data with a strong ROAS drop.
        ROAS = revenue / spend
        """
        rng = np.random.default_rng(42)

        # Baseline: higher revenue → higher ROAS
        baseline = pd.DataFrame({
            "spend": np.full(n, 100.0),
            "revenue": rng.normal(300.0, 20.0, n),  # mean ROAS ~ 3.0
            "impressions": rng.integers(800, 1200, n),
            "clicks": rng.integers(15, 30, n),
            "purchases": rng.integers(5, 15, n),
            "creative_type": ["Video"] * n,
            "audience_type": ["Broad"] * n,
        })

        # Current: lower revenue → strong ROAS drop
        current = pd.DataFrame({
            "spend": np.full(n, 100.0),
            "revenue": rng.normal(200.0, 20.0, n),  # mean ROAS ~ 2.0 (≈ -33%)
            "impressions": rng.integers(800, 1200, n),
            "clicks": rng.integers(15, 30, n),
            "purchases": rng.integers(5, 15, n),
            "creative_type": ["Video"] * n,
            "audience_type": ["Broad"] * n,
        })

        return baseline, current

    def _build_insight(
        self,
        hypothesis: str,
        metrics,
        segments=None,
        expected_direction="decrease",
        evidence_type="aggregated_performance",
        priority="high",
        confidence_hint=0.7,
    ) -> Insight:
        """Create a minimal, valid Insight object for Evaluator V2."""
        return Insight(
            hypothesis=hypothesis,
            driver="Test Driver",
            evidence_type=evidence_type,
            affected_metrics=list(metrics),
            affected_segments=segments or {},
            priority=priority,
            score=0.8,
            confidence_hint=confidence_hint,
            expected_direction=expected_direction,
            evidence=[],   # Evaluator recomputes from raw data
            metadata={},
        )

    # ------------------------------------------------------------------ #
    # Core behaviour tests
    # ------------------------------------------------------------------ #
    def test_significant_roas_drop_is_validated(self):
        """Strong ROAS drop should produce a validated result with decent confidence."""
        baseline, current = self._make_base_current_df_roas_drop(n=80)

        insight = self._build_insight(
            hypothesis="ROAS has dropped significantly globally",
            metrics=["roas"],
            segments={},  # global
            expected_direction="decrease",
            evidence_type="global_performance",
            priority="high",
            confidence_hint=0.7,
        )

        result = self.evaluator.evaluate_insight(
            insight=insight,
            baseline_df=baseline,
            current_df=current,
        )

        self.assertIsInstance(result, ValidationResult)
        self.assertEqual(result.hypothesis, insight.hypothesis)

        # We expect a strong drop → validated
        self.assertTrue(result.validated, "Expected hypothesis to be validated for strong ROAS drop")
        self.assertGreaterEqual(
            result.confidence,
            self.config["evaluator"]["confidence_minimum"],
            "Confidence should exceed configured minimum for strong effects",
        )

        # Evidence should include ROAS
        self.assertTrue(
            any(ev.metric == "roas" for ev in result.evidence),
            "Evidence should contain ROAS metric",
        )

    def test_no_change_not_validated_and_low_confidence(self):
        """When distributions are the same, validation should fail and confidence should stay low."""
        rng = np.random.default_rng(123)

        baseline = pd.DataFrame({
            "spend": np.full(80, 100.0),
            "revenue": rng.normal(300.0, 20.0, 80),
            "impressions": rng.integers(800, 1200, 80),
            "clicks": rng.integers(15, 30, 80),
            "purchases": rng.integers(5, 15, 80),
            "creative_type": ["Image"] * 80,
            "audience_type": ["Broad"] * 80,
        })

        # Current ~ same distribution as baseline
        current = baseline.copy()

        insight = self._build_insight(
            hypothesis="ROAS has changed",
            metrics=["roas"],
            segments={},
            expected_direction="decrease",
            evidence_type="global_performance",
            priority="medium",
            confidence_hint=0.5,
        )

        result = self.evaluator.evaluate_insight(
            insight=insight,
            baseline_df=baseline,
            current_df=current,
        )

        self.assertIsInstance(result, ValidationResult)
        self.assertFalse(result.validated, "Should not validate when distributions are effectively the same")
        self.assertLessEqual(
            result.confidence,
            0.7,
            "Confidence should not be very high when there is no real change",
        )

    def test_segment_filter_creative_type_applied(self):
        """If insight is for creative_type=Video, evaluator should filter correctly and annotate evidence."""
        rng = np.random.default_rng(999)

        # Mix of Video and Image in both periods
        n = 100
        creative_types = ["Video"] * (n // 2) + ["Image"] * (n // 2)

        baseline = pd.DataFrame({
            "spend": np.full(n, 100.0),
            "revenue": list(rng.normal(300.0, 20.0, n // 2)) + list(rng.normal(280.0, 20.0, n // 2)),
            "impressions": rng.integers(800, 1200, n),
            "clicks": rng.integers(15, 30, n),
            "purchases": rng.integers(5, 15, n),
            "creative_type": creative_types,
            "audience_type": ["Broad"] * n,
        })

        # Current: Video ROAS drops, Image stable
        current = pd.DataFrame({
            "spend": np.full(n, 100.0),
            "revenue": list(rng.normal(200.0, 20.0, n // 2)) + list(rng.normal(280.0, 20.0, n // 2)),
            "impressions": rng.integers(800, 1200, n),
            "clicks": rng.integers(15, 30, n),
            "purchases": rng.integers(5, 15, n),
            "creative_type": creative_types,
            "audience_type": ["Broad"] * n,
        })

        insight = self._build_insight(
            hypothesis="Video creatives underperforming on ROAS",
            metrics=["roas"],
            segments={"creative_type": "Video"},
            expected_direction="decrease",
            evidence_type="segment_drop",
            priority="high",
            confidence_hint=0.7,
        )

        result = self.evaluator.evaluate_insight(
            insight=insight,
            baseline_df=baseline,
            current_df=current,
        )

        self.assertIsInstance(result, ValidationResult)
        self.assertTrue(result.validated, "Expected Video segment to be validated for ROAS drop")

        # Evidence should reflect segment context
        self.assertTrue(
            any(ev.segment == "creative_type" and ev.segment_value == "Video" for ev in result.evidence),
            "Evidence should be tagged with segment='creative_type', value='Video'",
        )

    def test_nan_and_inf_are_handled_gracefully(self):
        """Evaluator should not crash when NaN / Inf values are present."""
        # We'll test on a raw metric that isn't derived (e.g., 'spend')
        # so we can inject NaNs / Infs directly.
        base = pd.DataFrame({
            "spend": [100.0, 200.0, np.nan, np.inf, 150.0, 130.0],
            "revenue": [300, 600, 0, 0, 450, 390],
            "impressions": [1000, 1200, 0, 0, 900, 950],
            "clicks": [20, 25, 0, 0, 18, 19],
            "purchases": [5, 7, 0, 0, 4, 3],
            "creative_type": ["Image"] * 6,
            "audience_type": ["Broad"] * 6,
        })

        curr = pd.DataFrame({
            "spend": [90.0, 210.0, np.nan, -np.inf, 140.0, 135.0],
            "revenue": [270, 630, 0, 0, 420, 400],
            "impressions": [1000, 1200, 0, 0, 900, 950],
            "clicks": [20, 25, 0, 0, 18, 19],
            "purchases": [5, 7, 0, 0, 4, 3],
            "creative_type": ["Image"] * 6,
            "audience_type": ["Broad"] * 6,
        })

        # For direct metric 'spend', evaluator won't use DERIVED_METRIC_FUNCS
        insight = self._build_insight(
            hypothesis="Spend changed",
            metrics=["spend"],
            segments={},
            expected_direction="increase",
            evidence_type="aggregated_performance",
            priority="medium",
            confidence_hint=0.5,
        )

        result = self.evaluator.evaluate_insight(
            insight=insight,
            baseline_df=base,
            current_df=curr,
        )

        self.assertIsInstance(result, ValidationResult)
        # Should not crash, and evidence may or may not validate depending on noise
        self.assertIsInstance(result.evidence, list)

    # ------------------------------------------------------------------ #
    # Dataclass shape tests
    # ------------------------------------------------------------------ #
    def test_evidence_to_dict_structure(self):
        """Evidence dataclass should serialize cleanly with bootstrap_ci normalized to plain floats."""
        ev = Evidence(
            metric="roas",
            baseline_value=3.0,
            current_value=2.0,
            absolute_delta=-1.0,
            relative_delta_pct=-33.3,
            sample_size_baseline=100,
            sample_size_current=100,
            p_value=0.001,
            adjusted_p_value=0.002,
            power=0.9,
            effect_size=-0.8,
            bootstrap_ci={"ci_lower": -1.2, "ci_upper": -0.8, "zero_included": False},
            drift_detected=False,
            segment="creative_type",
            segment_value="Video",
        )

        d = ev.to_dict()
        self.assertIn("metric", d)
        self.assertIn("baseline_value", d)
        self.assertIn("bootstrap_ci", d)
        self.assertIsInstance(d["bootstrap_ci"], dict)
        self.assertIn("ci_lower", d["bootstrap_ci"])

    def test_validation_result_to_dict_structure(self):
        """ValidationResult dataclass should serialize including nested Evidence."""
        ev = Evidence(
            metric="roas",
            baseline_value=3.0,
            current_value=2.0,
            absolute_delta=-1.0,
            relative_delta_pct=-33.3,
            sample_size_baseline=100,
            sample_size_current=100,
            p_value=0.001,
            adjusted_p_value=0.002,
            power=0.9,
            effect_size=-0.8,
            bootstrap_ci={},
            drift_detected=False,
        )

        vr = ValidationResult(
            hypothesis="Test hypothesis",
            validated=True,
            impact="high",
            confidence=0.85,
            evidence=[ev],
            statistical_tests={"roas": {"p_value": 0.001}},
            severity="severe",
            recommendation="Test recommendation",
            run_id="test_run",
            metadata={"method": "unit_test"},
        )

        d = vr.to_dict()
        self.assertIn("hypothesis", d)
        self.assertIn("validated", d)
        self.assertIn("evidence", d)
        self.assertIsInstance(d["evidence"], list)
        self.assertIsInstance(d["evidence"][0], dict)
        self.assertEqual(d["hypothesis"], "Test hypothesis")
        self.assertAlmostEqual(d["confidence"], 0.85, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
