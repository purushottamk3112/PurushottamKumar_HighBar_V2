"""
Creative Generator - Production Grade V2
Generates creative recommendations tightly linked to diagnosed issues.
Compatible with InsightAgent V2 + Evaluator V2.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from utils import ProductionLogger


# ======================================================
# Data Structure
# ======================================================
@dataclass
class CreativeRecommendation:
    """Structured creative recommendation"""
    title: str
    creative_type: str
    messaging_angle: str
    target_segment: Dict[str, str]
    rationale: str
    linked_insight: str
    expected_impact: str
    testing_strategy: str
    success_metrics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ======================================================
# Creative Generator V2
# ======================================================
class CreativeGenerator:
    """
    Production-level creative generator:
    - Uses validated insights only
    - Builds tightly-linked creative suggestions
    - Handles missing fields safely
    - Supports multiple orchestrator signatures
    """

    def __init__(self, config: Dict[str, Any], logger: ProductionLogger):
        self.config = config or {}
        self.logger = logger
        self.logger.info("CreativeGenerator V2 initialized")

    # ------------------------------------------------------------------
    # PUBLIC API — BOTH METHODS are supported for backward compatibility
    # ------------------------------------------------------------------
    def generate_recommendations(self, insights, validations):
        return self._generate(insights, validations)

    def generate_creatives(self, insights, validations):
        return self._generate(insights, validations)

    # ------------------------------------------------------------------
    # CORE DISPATCH
    # ------------------------------------------------------------------
    def _generate(self, insights, validations) -> List[Dict[str, Any]]:
        self.logger.info("Starting creative recommendation generation")

        try:
            if not insights or not validations:
                self.logger.warning("No insights or validations provided")
                return []

            # Map validated results for fast lookup
            val_map = {
                getattr(v, "hypothesis", None): v
                for v in validations
                if getattr(v, "validated", False)
            }

            validated_pairs = []
            for ins in insights:
                hypo = getattr(ins, "hypothesis", None)
                if hypo and hypo in val_map:
                    validated_pairs.append((ins, val_map[hypo]))

            if not validated_pairs:
                self.logger.warning("No validated insights available for creative generation")
                return []

            recommendations = []
            for insight, validation in validated_pairs:
                try:
                    recs = self._generate_for_insight(insight, validation)
                    recommendations.extend(recs)
                except Exception as e:
                    self.logger.error(
                        "Failed generating recommendations for insight",
                        insight=getattr(insight, "hypothesis", "")[:60],
                        exception=str(e),
                    )

            # Limit via config
            limit = int(self.config.get("agents", {}).get("creative_generator", {}).get("max_suggestions", 5))
            recommendations = recommendations[:limit]

            self.logger.log_output_summary(
                "CREATIVE_RECOMMENDATIONS",
                {
                    "total_recommendations": len(recommendations),
                    "linked_validations": len(validated_pairs),
                },
            )

            return [r.to_dict() for r in recommendations]

        except Exception as e:
            self.logger.error("Creative generation failed", exception=str(e))
            return []

    # ------------------------------------------------------------------
    # DISPATCH PER INSIGHT CATEGORY
    # ------------------------------------------------------------------
    def _generate_for_insight(self, insight, validation):
        e_type = getattr(insight, "evidence_type", None) or getattr(insight, "type", "metric_drop")
        recommendations = []

        if e_type == "creative_fatigue":
            recommendations.extend(self._recommendations_for_fatigue(insight, validation))

        elif e_type == "segment_drop":
            recommendations.extend(self._recommendations_for_segment_drop(insight, validation))

        elif e_type == "audience_shift":
            recommendations.extend(self._recommendations_for_audience_shift(insight, validation))

        elif e_type in ("global_performance", "aggregated_performance", "metric_drop"):
            recommendations.extend(self._recommendations_for_metric_drop(insight, validation))

        else:
            # Safe fallback
            recommendations.extend(self._recommendations_for_metric_drop(insight, validation))

        return recommendations

    # ======================================================
    # RECOMMENDATION BUILDERS
    # ======================================================
    def _recommendations_for_fatigue(self, insight, validation):
        segments = getattr(insight, "affected_segments", {}) or {}
        ctype = segments.get("creative_type", "Creative")
        evidence_str = self._format_evidence(validation)

        return [
            CreativeRecommendation(
                title=f"Combat Creative Fatigue in '{ctype}'",
                creative_type="UGC" if ctype.lower() == "video" else "Video",
                messaging_angle=f"Introduce pattern interrupts + new user-generated angles for {ctype}.",
                target_segment=segments,
                rationale=f"Creative fatigue validated ({evidence_str}). Metrics show deteriorating CTR under stable spend.",
                linked_insight=getattr(insight, "hypothesis", ""),
                expected_impact="high",
                testing_strategy="Test 3 refreshed creatives with new hooks; allocate 20% of spend.",
                success_metrics=["ctr", "engagement", "frequency"],
            )
        ]

    def _recommendations_for_segment_drop(self, insight, validation):
        segments = getattr(insight, "affected_segments", {}) or {}
        seg_str = ", ".join(f"{k}={v}" for k, v in segments.items()) or "segment"
        evidence_str = self._format_evidence(validation)

        return [
            CreativeRecommendation(
                title=f"Improve Performance for {seg_str}",
                creative_type="Static Image",
                messaging_angle=f"Segment-personalized messaging targeting {seg_str}.",
                target_segment=segments,
                rationale=f"Performance decline validated ({evidence_str}). Tailored creative expected to improve relevance.",
                linked_insight=getattr(insight, "hypothesis", ""),
                expected_impact="high",
                testing_strategy="Run A/B tests with segment-specific variants (~25% budget).",
                success_metrics=["roas", "cvr", "ctr"],
            )
        ]

    def _recommendations_for_audience_shift(self, insight, validation):
        segments = getattr(insight, "affected_segments", {}) or {}
        audience = segments.get("audience_type", "audience")
        evidence_str = self._format_evidence(validation)

        return [
            CreativeRecommendation(
                title=f"Broaden Appeal Beyond '{audience}'",
                creative_type="Carousel",
                messaging_angle="Social proof + multi-angle messaging to reach adjacent audiences.",
                target_segment={"audience_type": "expanded"},
                rationale=f"Audience underperformance confirmed ({evidence_str}). Expansion recommended.",
                linked_insight=getattr(insight, "hypothesis", ""),
                expected_impact="medium",
                testing_strategy="Run split tests: original audience vs blended LLA/interest groups.",
                success_metrics=["roas", "reach", "frequency"],
            )
        ]

    def _recommendations_for_metric_drop(self, insight, validation):
        metrics = getattr(insight, "affected_metrics", []) or []
        evidence_str = self._format_evidence(validation)
        recs = []

        if "ctr" in metrics:
            recs.append(
                CreativeRecommendation(
                    title="Boost CTR with Short-Form Video",
                    creative_type="Short Video",
                    messaging_angle="High-contrast hook + question-based opener.",
                    target_segment={},
                    rationale=f"CTR decline validated ({evidence_str}). Strong hook needed.",
                    linked_insight=getattr(insight, "hypothesis", ""),
                    expected_impact="high",
                    testing_strategy="Test 3 hooks; allocate 20% test budget.",
                    success_metrics=["ctr", "3s_view_rate", "clicks"],
                )
            )

        if "roas" in metrics or "revenue" in metrics:
            recs.append(
                CreativeRecommendation(
                    title="Drive Conversions with UGC + Offer Overlay",
                    creative_type="UGC / Collection",
                    messaging_angle="Social proof + urgency",
                    target_segment={},
                    rationale=f"ROAS deterioration validated ({evidence_str}).",
                    linked_insight=getattr(insight, "hypothesis", ""),
                    expected_impact="high",
                    testing_strategy="Test UGC vs best performer with offer overlay.",
                    success_metrics=["roas", "cvr", "revenue"],
                )
            )

        if not recs:  # Safe fallback
            recs.append(
                CreativeRecommendation(
                    title="General Creative Refresh",
                    creative_type="Static Image",
                    messaging_angle="New visuals + clear CTA",
                    target_segment={},
                    rationale=f"Metric decline validated ({evidence_str}).",
                    linked_insight=getattr(insight, "hypothesis", ""),
                    expected_impact="medium",
                    testing_strategy="Small-batch refresh of best creatives.",
                    success_metrics=["ctr", "roas"],
                )
            )

        return recs

    # ======================================================
    # UTILITIES
    # ======================================================
    def _format_evidence(self, validation):
        """Safely format evidence text for rationale strings."""
        try:
            ev_list = getattr(validation, "evidence", [])
            if not ev_list:
                return "validated decline"

            ev = ev_list[0]
            metric = getattr(ev, "metric", "metric").upper()
            pct = getattr(ev, "relative_delta_pct", None)

            if pct is None:
                return f"{metric} change"

            return f"{metric} {pct:+.1f}%"

        except Exception:
            return "validated trend"


if __name__ == "__main__":
    print("CreativeGenerator V2 Loaded")
