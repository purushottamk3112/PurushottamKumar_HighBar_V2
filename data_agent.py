import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import timedelta

from utils import ProductionLogger, safe_divide

# Optional schema imports
try:
    from schema import DataSchema, validate_and_clean_data
except Exception:
    DataSchema = None

    def validate_and_clean_data(df, schema):
        return df.copy(), {"valid": True, "errors": [], "warnings": []}


class DataAgent:
    """
    DataAgent V2 – Production Grade (Stateless)

    Responsibilities:
    - Load CSV
    - Validate schema
    - Clean numeric fields
    - Compute baseline/current windows
    - Provide orchestrator-friendly methods
    """

    REQUIRED_COLUMNS = {
        "date", "spend", "revenue",
        "impressions", "clicks", "purchases",
        "ctr", "roas"
    }

    def __init__(self, config: Dict[str, Any], logger: ProductionLogger):
        self.config = config
        self.logger = logger
        self.schema = DataSchema() if DataSchema else None
        self._last_df = None

        self.logger.info("DataAgent V2 initialized")

    # ======================================================================
    # LOAD + VALIDATE
    # ======================================================================
    def load_and_validate(self, csv_path: str) -> Dict[str, Any]:
        """Load CSV and apply schema validation + cleaning."""
        try:
            df = self._load_csv_internal(csv_path)
            cleaned_df, validation = self._validate_and_clean(df)

            return {
                "success": validation["valid"],
                "data": cleaned_df,
                "errors": validation.get("errors", []),
                "warnings": validation.get("warnings", [])
            }
        except Exception as e:
            return {"success": False, "errors": [str(e)]}

    # ------------------------------------------------------------------
    # REQUIRED BY ORCHESTRATOR — PUBLIC SCHEMA VALIDATION WRAPPER
    # ------------------------------------------------------------------
    def validate_schema(self, df: pd.DataFrame):
        """
        Public schema validation API (required by Orchestrator).
        Returns: dict(valid=bool, errors=[...], warnings=[...])
        """
        _, validation = self._validate_and_clean(df)
        return validation

    # ======================================================================
    # INTERNAL CSV LOADER
    # ======================================================================
    def _load_csv_internal(self, csv_path: str) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(path)

        # Parse date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Enforce numeric columns before schema
        for col in ["spend", "revenue", "clicks", "impressions", "purchases"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        self._last_df = df.copy()

        self.logger.log_input_summary("LOAD_CSV", {
            "path": str(path),
            "rows": len(df),
            "columns": list(df.columns)
        })

        return df

    # ======================================================================
    # SCHEMA VALIDATION + CLEANING
    # ======================================================================
    def _validate_and_clean(self, df: pd.DataFrame):
        """Ensures required columns + schema cleaning."""
        missing = list(self.REQUIRED_COLUMNS - set(df.columns))
        strict = self.config.get("schema", {}).get("strict_mode", True)

        # Required column check
        if missing and strict:
            return df, {
                "valid": False,
                "errors": [f"Missing required columns: {missing}"],
                "warnings": []
            }

        if missing and not strict:
            self.logger.warning("Missing required columns", missing=missing)

        # If schema exists → use advanced validator
        if DataSchema:
            cleaned_df, validation = validate_and_clean_data(df.copy(), self.schema)
            self._last_df = cleaned_df.copy()
            return cleaned_df, validation

        # Fallback (no schema)
        return df, {"valid": True, "errors": [], "warnings": []}

    # ======================================================================
    # BASELINE + CURRENT SPLIT
    # ======================================================================
    def split_baseline_current(self, df: Optional[pd.DataFrame] = None):
        """Return baseline_df, current_df, metadata."""
        if df is None:
            df = self._last_df
            if df is None:
                raise ValueError("No DataFrame provided and no loaded data available.")
        return self._safe_split_logic(df)

    def compute_baseline_and_current(self):
        """Compatibility wrapper for orchestrator."""
        if self._last_df is None:
            return {"success": False, "error": "No data loaded"}

        try:
            base, curr, meta = self._safe_split_logic(self._last_df)
            return {
                "success": True,
                "baseline_data": base,
                "current_data": curr,
                "metadata": meta
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------
    # SAFE SPLIT LOGIC
    # ------------------------------------------------------------------
    def _safe_split_logic(self, df):
        if "date" not in df.columns:
            raise ValueError("Missing required 'date' column")

        cfg = self.config.get("analysis", {})
        baseline_days = int(cfg.get("baseline_days", 14))
        current_days = int(cfg.get("current_days", 7))
        min_samples = int(cfg.get("min_sample_size", 10))

        max_dt = df["date"].max()
        if pd.isna(max_dt):
            raise ValueError("Invalid date values in dataset")

        current_start = max_dt - timedelta(days=current_days - 1)
        baseline_start = current_start - timedelta(days=baseline_days)

        baseline_df = df[(df["date"] >= baseline_start) & (df["date"] < current_start)].copy()
        current_df = df[df["date"] >= current_start].copy()

        # Safety logs
        if len(baseline_df) < min_samples:
            self.logger.warning("Baseline window too small", rows=len(baseline_df))

        if len(current_df) < min_samples:
            self.logger.warning("Current window too small", rows=len(current_df))

        metadata = {
            "baseline_days": baseline_days,
            "current_days": current_days,
            "baseline_rows": len(baseline_df),
            "current_rows": len(current_df),
            "baseline_start": baseline_start.strftime("%Y-%m-%d"),
            "current_start": current_start.strftime("%Y-%m-%d"),
            "max_date": max_dt.strftime("%Y-%m-%d")
        }

        return baseline_df, current_df, metadata

    # ======================================================================
    # WRAPPERS
    # ======================================================================
    def load_csv(self, csv_path: str):
        """Direct access load method."""
        return self._load_csv_internal(csv_path)

    def load_and_split_data(self, csv_path: str):
        """Load CSV → validate → split windows."""
        df = self.load_csv(csv_path)
        cleaned, validation = self._validate_and_clean(df)

        if not validation["valid"]:
            self.logger.warning("Schema validation warnings", issues=validation.get("errors"))

        base, curr, meta = self._safe_split_logic(cleaned)
        return cleaned, base, curr, meta


if __name__ == "__main__":
    from utils import ConfigLoader, ProductionLogger
    config = ConfigLoader.load("config.yaml")
    logger = ProductionLogger("DataAgentTest", "logs")
    agent = DataAgent(config, logger)
    res = agent.load_and_validate(config["data"]["input_csv"])
    print(res)
