"""
Orchestrator V2 – Production Grade
Coordinates the complete stateless pipeline:
DataAgent → InsightAgent → Evaluator → CreativeGenerator
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from utils import ProductionLogger, ConfigLoader
from data_agent import DataAgent
from insight_agent import InsightAgent
from evaluator import Evaluator
from creative_generator import CreativeGenerator


class Orchestrator:
    """
    Production Orchestrator V2
    Fully stateless, deterministic, robust, and High-Bar compliant.
    """

    def __init__(self, config_path: str = "config.yaml"):
        # Load config
        self.config = ConfigLoader.load(config_path)

        # Generate run id + directory
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_root = Path(self.config["data"].get("output_dir", "outputs"))
        self.run_dir = output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        log_dir = self.run_dir / "logs"
        self.logger = ProductionLogger("Orchestrator", str(log_dir), self.run_id)

        self.logger.info("=" * 80)
        self.logger.info(" KASPARRO V2 – PRODUCTION SYSTEM ")
        self.logger.info("=" * 80)
        self.logger.info(f"Run Directory: {self.run_dir}")

        # Initialize agents
        self.data_agent = DataAgent(
            self.config,
            ProductionLogger("DataAgent", str(log_dir), self.run_id)
        )
        self.insight_agent = InsightAgent(
            self.config,
            ProductionLogger("InsightAgent", str(log_dir), self.run_id)
        )
        self.evaluator = Evaluator(
            self.config,
            ProductionLogger("Evaluator", str(log_dir), self.run_id)
        )
        self.creative_generator = CreativeGenerator(
            self.config,
            ProductionLogger("CreativeGenerator", str(log_dir), self.run_id)
        )

        self.logger.info("All agents initialized.\n")

    # ======================================================================
    # MAIN PIPELINE
    # ======================================================================
    def run(self, csv_path: str = None) -> Dict[str, Any]:
        result = {
            "success": False,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "timestamp": datetime.now().isoformat(),
            "errors": []
        }

        try:
            # ------------------------------------------------------------------
            # STEP 1 — LOAD CSV
            # ------------------------------------------------------------------
            self.logger.info("STEP 1: Loading CSV")

            csv_path = csv_path or self.config["data"]["input_csv"]
            if not csv_path:
                raise ValueError("No CSV path provided")

            df = self.data_agent.load_csv(csv_path)

            self.logger.log_input_summary("RAW_DATA", {
                "rows": len(df),
                "columns": list(df.columns)
            })

            # ------------------------------------------------------------------
            # STEP 2 — SCHEMA VALIDATION
            # ------------------------------------------------------------------
            self.logger.info("STEP 2: Validating Schema")

            schema_validation = self.data_agent.validate_schema(df)
            if not schema_validation["valid"]:
                raise ValueError(f"Schema validation failed: {schema_validation['errors']}")

            # ------------------------------------------------------------------
            # STEP 3 — BASELINE & CURRENT SPLIT
            # ------------------------------------------------------------------
            self.logger.info("STEP 3: Splitting baseline and current data")

            baseline_df, current_df, metadata = self.data_agent.split_baseline_current(df)

            self.logger.log_output_summary("PERIOD_SPLIT", {
                "baseline_rows": len(baseline_df),
                "current_rows": len(current_df)
            })

            # ------------------------------------------------------------------
            # STEP 4 — INSIGHT GENERATION
            # ------------------------------------------------------------------
            self.logger.info("STEP 4: Generating insights")

            insights = self.insight_agent.generate_insights(baseline_df, current_df)
            self.logger.info(f"Generated {len(insights)} insights.")

            # ------------------------------------------------------------------
            # STEP 5 — VALIDATION (Evaluator V2)
            # ------------------------------------------------------------------
            self.logger.info("STEP 5: Validating insights")

            validations = []
            validated_count = 0

            for insight in insights:
                try:
                    metrics = getattr(insight, "affected_metrics", []) or []
                    # Fallback if somehow empty
                    if not metrics:
                        metrics = ["roas", "ctr"]

                    segments = getattr(insight, "affected_segments", {}) or {}
                    expected_direction = getattr(insight, "expected_direction", "decrease")

                    v = self.evaluator.evaluate_hypothesis(
                        hypothesis=insight.hypothesis,
                        baseline_data=baseline_df,
                        current_data=current_df,
                        metrics=metrics,
                        segments=segments if segments else None,
                        expected_direction=expected_direction,
                    )
                    validations.append(v)

                    if v.validated:
                        validated_count += 1
                        self.logger.info(
                            f"✓ VALID: {insight.hypothesis[:80]}...",
                            confidence=v.confidence,
                            impact=v.impact
                        )
                    else:
                        self.logger.info(
                            f"✗ NOT VALID: {insight.hypothesis[:80]}...",
                            confidence=v.confidence
                        )
                except Exception as e:
                    self.logger.error(
                        f"Validation failed for insight: {getattr(insight, 'hypothesis', '')[:80]}",
                        exception=str(e)
                    )

            self.logger.info(f"Validated {validated_count}/{len(insights)} insights.")

            # ------------------------------------------------------------------
            # STEP 6 — CREATIVE GENERATION
            # ------------------------------------------------------------------
            self.logger.info("STEP 6: Generating creative recommendations")

            recommendations = self.creative_generator.generate_creatives(
                insights=insights,
                validations=validations
            )

            self.logger.info(f"Generated {len(recommendations)} recommendations.")

            # ------------------------------------------------------------------
            # STEP 7 — OUTPUT FILES
            # ------------------------------------------------------------------
            self.logger.info("STEP 7: Generating outputs")

            self._generate_outputs(insights, validations, recommendations, metadata)

            result.update({
                "success": True,
                "insights_generated": len(insights),
                "insights_validated": validated_count,
                "recommendations_generated": len(recommendations)
            })

            self.logger.info("=" * 80)
            self.logger.info(" ANALYSIS COMPLETE ")
            self.logger.info("=" * 80)

            return result

        except Exception as e:
            err = f"Pipeline crashed: {str(e)}"
            self.logger.error(err)
            result["errors"].append(err)
            return result

    # ======================================================================
    # OUTPUT GENERATION
    # ======================================================================
    def _generate_outputs(self, insights, validations, recommendations, metadata):
        reports_dir = self.run_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        # -----------------------------
        # INSIGHTS JSON
        # -----------------------------
        insights_json = {
            "metadata": metadata,
            "insights": []
        }

        for ins in insights:
            ins_dict = ins.to_dict()
            match = next((v for v in validations if v.hypothesis == ins.hypothesis), None)
            if match:
                ins_dict["validation"] = match.to_dict()
            insights_json["insights"].append(ins_dict)

        with open(reports_dir / "insights.json", "w") as f:
            json.dump(insights_json, f, indent=2)

        # -----------------------------
        # RECOMMENDATIONS JSON
        # -----------------------------
        with open(reports_dir / "recommendations.json", "w") as f:
            json.dump({
                "metadata": metadata,
                "recommendations": recommendations
            }, f, indent=2)

        # -----------------------------
        # MARKDOWN REPORT
        # -----------------------------
        self._generate_markdown_report(
            insights, validations, recommendations, metadata, reports_dir / "report.md"
        )

    # ======================================================================
    # MARKDOWN REPORT
    # ======================================================================
    def _generate_markdown_report(self, insights, validations, recommendations, metadata, output_path):

        validated = [v for v in validations if v.validated]

        lines = [
            "# Facebook Ads Performance Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Executive Summary",
            "",
            f"- Total Insights: {len(insights)}",
            f"- Validated Insights: {len(validated)}",
            f"- Creative Recommendations: {len(recommendations)}",
            "",
            "## Key Findings",
            ""
        ]

        for v in validated[:5]:
            lines.append(f"### {v.hypothesis}")
            lines.append(f"- Impact: **{v.impact}**")
            lines.append(f"- Confidence: **{v.confidence:.0%}**")
            lines.append(f"- Severity: **{v.severity}**")
            lines.append("")
            lines.append("**Evidence:**")
            for e in v.evidence:
                lines.append(
                    f"- {e.metric}: {e.relative_delta_pct:+.2f}% "
                    f"(Base={e.baseline_value:.4f}, Curr={e.current_value:.4f})"
                )
            lines.append("")

        lines.append("## Creative Recommendations\n")

        for rec in recommendations:
            title = rec.get("title", "Recommendation")
            lines.append(f"### {title}")
            for k, v in rec.items():
                if k != "title":
                    lines.append(f"- **{k.replace('_',' ').title()}:** {v}")
            lines.append("")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))


if __name__ == "__main__":
    orchestrator = Orchestrator("config.yaml")
    output = orchestrator.run()
    print(output)
