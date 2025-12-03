import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from utils import ProductionLogger, safe_divide, safe_percentage_change


# =========================================================
# Data Structures
# =========================================================
@dataclass
class InsightEvidence:
    metric: str
    baseline_value: float
    current_value: float
    absolute_delta: float
    relative_delta_pct: float
    sample_size_baseline: int
    sample_size_current: int
    segment: Optional[str] = None
    segment_value: Optional[str] = None
    signal_strength: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Insight:
    hypothesis: str
    driver: str
    evidence_type: str
    affected_metrics: List[str]
    affected_segments: Dict[str, str]
    priority: str
    score: float
    confidence_hint: float
    expected_direction: str
    evidence: List[InsightEvidence]
    metadata: Dict[str, Any]
    needs_validation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d


# =========================================================
# InsightAgent V2 – Production Grade
# =========================================================
class InsightAgent:
    REQUIRED_COLUMNS = {
        "date",
        "spend",
        "revenue",
        "clicks",
        "impressions",
        "purchases",
        "ctr",
        "roas",
    }

    def __init__(self, config: Dict[str, Any], logger: ProductionLogger):
        self.config = config or {}
        self.logger = logger

        # ---------------- Thresholds ----------------
        t = self.config.get("thresholds", {})

        # Positive metrics – drops are bad
        self.roas_drop_severe_pct = t.get("roas_drop_severe", 0.30) * 100
        self.roas_drop_moderate_pct = t.get("roas_drop_moderate", 0.15) * 100
        self.ctr_drop_severe_pct = t.get("ctr_drop_severe", 0.25) * 100
        self.ctr_drop_moderate_pct = t.get("ctr_drop_moderate", 0.10) * 100

        # Negative metrics – increases are bad
        self.cpa_increase_limit = t.get("audience_cpa_increase_pct", 20.0)

        # Global performance threshold
        self.global_drop_threshold = t.get("global_candidate_drop_pct", 10.0)

        # Creative fatigue thresholds
        self.fatigue_ctr_drop = t.get("creative_ctr_drop_pct", 15.0)
        self.fatigue_spend_growth = t.get("fatigue_spend_growth_pct", -5.0)

        # Guardrails
        self.min_spend = t.get("segment_min_spend", 50.0)
        self.min_spend_share = t.get("min_spend_share", 0.05)
        self.stability_lower = t.get("stability_lower_bound", 0.5)
        self.stability_upper = t.get("stability_upper_bound", 2.0)
        self.min_segment_sample = t.get("min_segment_sample", 5)

        # Segment dimensions
        self.dimensions = self.config.get("analysis", {}).get(
            "segment_dimensions",
            ["creative_type", "audience_type", "platform", "country", "placement"],
        )

        # Agent-level controls
        ia = self.config.get("agents", {}).get("insight_agent", {})
        self.max_hypotheses = ia.get("max_hypotheses", 7)
        self.min_hypotheses = ia.get("min_hypotheses", 3)
        self.require_evidence = ia.get("require_evidence", True)

        self.logger.info("InsightAgent V2 initialized")

    # =====================================================
    # Public API
    # =====================================================
    def generate_insights(
        self, baseline_data: pd.DataFrame, current_data: pd.DataFrame
    ) -> List[Insight]:
        """Main entrypoint: builds Insight objects from baseline/current data."""
        try:
            self._validate_schema(baseline_data, current_data)
        except Exception as e:
            if self.config.get("schema", {}).get("strict_mode", True):
                self.logger.error(f"Schema failure: {e}")
                return []
            self.logger.warning(f"Schema warning (non-strict): {e}")

        candidates: Dict[Any, List[InsightEvidence]] = defaultdict(list)

        # Stage 1: Global-level issues
        self._add_global_candidates(candidates, baseline_data, current_data)
        # Stage 2: Segment-level issues
        self._add_segment_candidates(candidates, baseline_data, current_data)
        # Stage 3: Creative fatigue patterns
        self._add_fatigue_candidates(candidates, baseline_data, current_data)
        # Stage 4: Audience shifts / CPA spikes
        self._add_audience_candidates(candidates, baseline_data, current_data)

        insights: List[Insight] = []

        for (dim, seg_val), ev_list in candidates.items():
            insight = self._synthesize_insight(dim, seg_val, ev_list)
            if insight:
                insights.append(insight)

        # Sort: priority (high→low), then score (desc)
        order = {"high": 0, "medium": 1, "low": 2}
        insights.sort(key=lambda x: (order.get(x.priority, 3), -x.score))

        return insights[: self.max_hypotheses]

    # =====================================================
    # Candidate Generators
    # =====================================================
    def _add_global_candidates(self, C, b_df: pd.DataFrame, c_df: pd.DataFrame):
        if b_df.empty or c_df.empty:
            return

        b = self._aggregate(b_df)
        c = self._aggregate(c_df)

        for m in ["roas", "ctr", "cvr", "cpa"]:
            base_val = b.get(m)
            curr_val = c.get(m)
            change = safe_percentage_change(base_val, curr_val)
            if change is None:
                continue

            if m == "cpa":
                is_bad = change > self.global_drop_threshold
            else:
                is_bad = change < -self.global_drop_threshold

            if not is_bad:
                continue

            C[("global", "all")].append(
                InsightEvidence(
                    metric=m,
                    baseline_value=base_val,
                    current_value=curr_val,
                    absolute_delta=curr_val - base_val,
                    relative_delta_pct=change,
                    sample_size_baseline=len(b_df),
                    sample_size_current=len(c_df),
                    segment="global",
                    segment_value="all",
                    signal_strength=abs(change) / 100.0,
                )
            )

    def _add_segment_candidates(self, C, b_df: pd.DataFrame, c_df: pd.DataFrame):
        for dim in self.dimensions:
            if dim not in b_df.columns or dim not in c_df.columns:
                continue

            common_vals = set(b_df[dim].dropna().unique()) & set(
                c_df[dim].dropna().unique()
            )
            if not common_vals:
                continue

            for seg in common_vals:
                b = b_df[b_df[dim] == seg]
                c = c_df[c_df[dim] == seg]

                if len(b) < self.min_segment_sample or len(c) < self.min_segment_sample:
                    continue

                if b["spend"].sum() < self.min_spend:
                    continue

                agg_b = self._aggregate(b)
                agg_c = self._aggregate(c)

                for m in ["roas", "ctr", "cpa"]:
                    base_val = agg_b.get(m)
                    curr_val = agg_c.get(m)
                    ch = safe_percentage_change(base_val, curr_val)
                    if ch is None:
                        continue

                    if m == "cpa":
                        is_bad = ch > self.cpa_increase_limit
                    else:
                        is_bad = ch < -self.roas_drop_moderate_pct

                    if not is_bad:
                        continue

                    C[(dim, str(seg))].append(
                        InsightEvidence(
                            metric=m,
                            baseline_value=base_val,
                            current_value=curr_val,
                            absolute_delta=curr_val - base_val,
                            relative_delta_pct=ch,
                            sample_size_baseline=len(b),
                            sample_size_current=len(c),
                            segment=dim,
                            segment_value=str(seg),
                            signal_strength=abs(ch) / 100.0,
                        )
                    )

    def _add_fatigue_candidates(self, C, b_df: pd.DataFrame, c_df: pd.DataFrame):
        if "creative_type" not in b_df.columns or "creative_type" not in c_df.columns:
            return

        types = set(c_df["creative_type"].dropna().unique())
        if not types:
            return

        for ctype in types:
            b = b_df[b_df["creative_type"] == ctype]
            c = c_df[c_df["creative_type"] == ctype]

            if len(b) < self.min_segment_sample or len(c) < self.min_segment_sample:
                continue

            b_ctr = safe_divide(b["clicks"].sum(), b["impressions"].sum())
            c_ctr = safe_divide(c["clicks"].sum(), c["impressions"].sum())
            ctr_ch = safe_percentage_change(b_ctr, c_ctr)

            b_spend = b["spend"].sum()
            c_spend = c["spend"].sum()
            spend_ch = safe_percentage_change(b_spend, c_spend)

            if (
                ctr_ch is not None
                and spend_ch is not None
                and ctr_ch < -self.fatigue_ctr_drop
                and spend_ch > self.fatigue_spend_growth
            ):
                C[("creative_fatigue", str(ctype))].append(
                    InsightEvidence(
                        metric="ctr",
                        baseline_value=b_ctr,
                        current_value=c_ctr,
                        absolute_delta=c_ctr - b_ctr,
                        relative_delta_pct=ctr_ch,
                        sample_size_baseline=len(b),
                        sample_size_current=len(c),
                        segment="creative_type",
                        segment_value=str(ctype),
                        signal_strength=abs(ctr_ch) / 100.0,
                    )
                )

    def _add_audience_candidates(self, C, b_df: pd.DataFrame, c_df: pd.DataFrame):
        if "audience_type" not in b_df.columns or "audience_type" not in c_df.columns:
            return

        total_spend = c_df["spend"].sum()
        if total_spend <= 0:
            return

        audiences = set(c_df["audience_type"].dropna().unique())
        if not audiences:
            return

        for aud in audiences:
            b = b_df[b_df["audience_type"] == aud]
            c = c_df[c_df["audience_type"] == aud]

            if len(b) < self.min_segment_sample or len(c) < self.min_segment_sample:
                continue

            b_cpa = safe_divide(b["spend"].sum(), b["purchases"].sum())
            c_cpa = safe_divide(c["spend"].sum(), c["purchases"].sum())
            ch = safe_percentage_change(b_cpa, c_cpa)

            spend_share = safe_divide(c["spend"].sum(), total_spend)

            if (
                ch is not None
                and ch > self.cpa_increase_limit
                and spend_share > self.min_spend_share
            ):
                C[("audience_shift", str(aud))].append(
                    InsightEvidence(
                        metric="cpa",
                        baseline_value=b_cpa,
                        current_value=c_cpa,
                        absolute_delta=c_cpa - b_cpa,
                        relative_delta_pct=ch,
                        sample_size_baseline=len(b),
                        sample_size_current=len(c),
                        segment="audience_type",
                        segment_value=str(aud),
                        signal_strength=(abs(ch) / 100.0) * spend_share,
                    )
                )

    # =====================================================
    # Synthesis
    # =====================================================
    def _synthesize_insight(
        self, dim: str, seg_val: str, ev_list: List[InsightEvidence]
    ) -> Optional[Insight]:
        # Only keep “bad” performance evidence
        perf = [
            e
            for e in ev_list
            if (e.metric == "cpa" and e.relative_delta_pct > 0)
            or (e.metric != "cpa" and e.relative_delta_pct < 0)
        ]

        if not perf:
            return None

        metrics = sorted({e.metric for e in perf})
        avg_sev = float(np.mean([abs(e.relative_delta_pct) for e in perf]))

        # Stability check
        ref = perf[0]
        ratio = safe_divide(ref.sample_size_current, ref.sample_size_baseline)
        unstable = ratio < self.stability_lower or ratio > self.stability_upper

        # Score: combine magnitude, multi-metric, sample, stability
        score = min(avg_sev / 50.0, 0.5)
        if len(metrics) > 1:
            score += 0.2
        if ref.sample_size_current > 50:
            score += 0.1
        if unstable:
            score -= 0.15

        score = max(0.0, min(score, 1.0))

        if score > 0.6:
            priority = "high"
        elif score > 0.35:
            priority = "medium"
        else:
            priority = "low"

        main_metric = sorted(
            perf, key=lambda x: abs(x.relative_delta_pct), reverse=True
        )[0].metric

        expected_dir = "increase" if main_metric == "cpa" else "decrease"

        # Narrative + evidence_type + segment mapping
        if dim == "global":
            e_type = "global_performance"
            hypo = f"Global performance deterioration detected in: {', '.join(metrics)}."
            driver = "Systemic Drop"
            affected_segments: Dict[str, str] = {}
        elif dim == "creative_fatigue":
            e_type = "creative_fatigue"
            hypo = f"Creative fatigue in type '{seg_val}' – CTR dropping under relatively stable spend."
            driver = "Creative Fatigue"
            affected_segments = {"creative_type": str(seg_val)}
        elif dim == "audience_shift":
            e_type = "audience_shift"
            hypo = f"Audience '{seg_val}' showing CPA surge and declining efficiency."
            driver = "Audience Saturation"
            affected_segments = {"audience_type": str(seg_val)}
        else:
            e_type = "segment_drop"
            hypo = f"Segment '{seg_val}' ({dim}) underperforming across {', '.join(metrics)}."
            driver = f"Segment Performance: {dim}"
            affected_segments = {dim: str(seg_val)}

        confidence_hint = 0.6 + (0.1 if len(metrics) > 1 else 0.0) - (
            0.2 if unstable else 0.0
        )
        confidence_hint = max(0.0, min(round(confidence_hint, 2), 1.0))

        return Insight(
            hypothesis=hypo,
            driver=driver,
            evidence_type=e_type,
            affected_metrics=metrics,
            affected_segments=affected_segments,
            priority=priority,
            score=round(score, 2),
            confidence_hint=confidence_hint,
            expected_direction=expected_dir,
            evidence=perf,
            metadata={
                "avg_severity_pct": round(avg_sev, 2),
                "sample_ratio": ratio,
                "stability": "unstable" if unstable else "stable",
                "evidence_count": len(perf),
            },
        )

    # =====================================================
    # Helpers
    # =====================================================
    def _aggregate(self, df: pd.DataFrame) -> Dict[str, float]:
        s = df["spend"].sum()
        r = df["revenue"].sum()
        c = df["clicks"].sum()
        i = df["impressions"].sum()
        p = df["purchases"].sum()
        return {
            "spend": s,
            "roas": safe_divide(r, s),
            "ctr": safe_divide(c, i),
            "cvr": safe_divide(p, c),
            "cpa": safe_divide(s, p),
        }

    def _validate_schema(self, b: pd.DataFrame, c: pd.DataFrame):
        missing_b = InsightAgent.REQUIRED_COLUMNS - set(b.columns)
        missing_c = InsightAgent.REQUIRED_COLUMNS - set(c.columns)
        if missing_b:
            raise ValueError(f"Baseline missing columns: {missing_b}")
        if missing_c:
            raise ValueError(f"Current missing columns: {missing_c}")


if __name__ == "__main__":
    print("InsightAgent V2 Loaded")
