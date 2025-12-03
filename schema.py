"""
Schema Definition and Validation – V2 High-Bar
Compatible with DataAgent V2 (stateless) and full V2 pipeline
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np


class ColumnType(Enum):
    STRING = "string"
    NUMERIC = "numeric"
    DATE = "date"
    CATEGORICAL = "categorical"


@dataclass
class ColumnSchema:
    """Schema definition for a single column"""
    name: str
    dtype: ColumnType
    required: bool = True
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[str]] = None
    
    def validate(self, series: pd.Series) -> List[str]:
        errors = []

        if series is None:
            if self.required:
                errors.append(f"Column '{self.name}' is required but missing.")
            return errors

        # Null validation
        if not self.nullable and series.isna().any():
            errors.append(
                f"Column '{self.name}' contains {series.isna().sum()} null values."
            )

        non_null = series.dropna()
        if len(non_null) == 0:
            return errors

        # Type validation
        if self.dtype == ColumnType.NUMERIC:
            numeric = pd.to_numeric(non_null, errors="coerce")
            if numeric.isna().any():
                errors.append(
                    f"Column '{self.name}' has {numeric.isna().sum()} invalid numeric values."
                )
            if self.min_value is not None:
                below = (numeric < self.min_value).sum()
                if below > 0:
                    errors.append(
                        f"Column '{self.name}' has {below} entries below min={self.min_value}"
                    )
            if self.max_value is not None:
                above = (numeric > self.max_value).sum()
                if above > 0:
                    errors.append(
                        f"Column '{self.name}' has {above} entries above max={self.max_value}"
                    )

        elif self.dtype == ColumnType.DATE:
            parsed = pd.to_datetime(non_null, errors="coerce")
            if parsed.isna().any():
                errors.append(f"Column '{self.name}' has invalid date values.")

        elif self.dtype == ColumnType.CATEGORICAL:
            if self.allowed_values:
                diff = set(non_null.astype(str)) - set(self.allowed_values)
                if diff:
                    errors.append(
                        f"Column '{self.name}' contains invalid categories: {diff}"
                    )

        return errors


@dataclass
class DataSchema:
    """High-bar schema for Facebook Ads data"""
    columns: List[ColumnSchema] = field(default_factory=list)

    def __post_init__(self):
        if not self.columns:
            self.columns = [
                ColumnSchema("campaign_name", ColumnType.STRING),
                ColumnSchema("adset_name", ColumnType.STRING),
                ColumnSchema("ad_name", ColumnType.STRING, required=False),

                ColumnSchema("date", ColumnType.DATE),

                ColumnSchema("spend", ColumnType.NUMERIC, min_value=0),
                ColumnSchema("impressions", ColumnType.NUMERIC, min_value=0),
                ColumnSchema("clicks", ColumnType.NUMERIC, min_value=0),
                ColumnSchema("ctr", ColumnType.NUMERIC, min_value=0, max_value=1),
                ColumnSchema("purchases", ColumnType.NUMERIC, min_value=0, nullable=True),
                ColumnSchema("revenue", ColumnType.NUMERIC, min_value=0, nullable=True),
                ColumnSchema("roas", ColumnType.NUMERIC, min_value=0, nullable=True),
                ColumnSchema("leads", ColumnType.NUMERIC, required=False, min_value=0, nullable=True),

                ColumnSchema(
                    "creative_type",
                    ColumnType.CATEGORICAL,
                    allowed_values=["Image", "Video", "Carousel", "UGC", "Collection"]
                ),
                ColumnSchema("creative_message", ColumnType.STRING, required=False),

                ColumnSchema(
                    "audience_type",
                    ColumnType.CATEGORICAL,
                    required=False,
                    allowed_values=["Broad", "Lookalike", "Retargeting", "Interest"]
                ),
                ColumnSchema(
                    "platform",
                    ColumnType.CATEGORICAL,
                    required=False,
                    allowed_values=["Facebook", "Instagram", "Audience Network", "Messenger"]
                ),
                ColumnSchema(
                    "country",
                    ColumnType.CATEGORICAL,
                    required=False,
                    allowed_values=["US", "UK", "IN", "CA", "AU"]
                ),
            ]

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns_validated": []
        }

        if df is None or df.empty:
            result["valid"] = False
            result["errors"].append("Dataframe is empty.")
            return result

        df_cols = set(df.columns)
        required_cols = {c.name for c in self.columns if c.required}

        missing = required_cols - df_cols
        if missing:
            result["valid"] = False
            result["errors"].append(f"Missing required columns: {missing}")
            return result

        # Validate each schema column
        for col in self.columns:
            if col.name in df.columns:
                errs = col.validate(df[col.name])
                if errs:
                    result["valid"] = False
                    result["errors"].extend(errs)

                result["columns_validated"].append(col.name)
            else:
                result["warnings"].append(f"Optional column '{col.name}' missing.")

        # Warn about extra columns
        extra = df_cols - {c.name for c in self.columns}
        if extra:
            result["warnings"].append(f"Ignoring extra columns: {extra}")

        return result

    # -------------------------
    # REQUIRED MISSING METHOD
    # -------------------------
    def get_required_columns(self) -> List[str]:
        return [c.name for c in self.columns if c.required]


# =============================================================================
# VALIDATE + CLEAN (used by DataAgent V2)
# =============================================================================

def validate_and_clean_data(df: pd.DataFrame, schema: DataSchema):
    validation = schema.validate_dataframe(df)

    if not validation["valid"]:
        return df, validation

    cleaned = df.copy()

    # Parse dates
    if "date" in cleaned.columns:
        cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")

    numeric_cols = [
        "spend", "impressions", "clicks", "ctr",
        "purchases", "revenue", "roas", "leads"
    ]

    for col in numeric_cols:
        if col not in cleaned.columns:
            cleaned[col] = 0.0
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").fillna(0.0)

    # Ensure categorical fallback
    if "creative_type" not in cleaned.columns:
        cleaned["creative_type"] = "Unknown"

    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    return cleaned, validation


if __name__ == "__main__":
    schema = DataSchema()
    print("Required columns:", schema.get_required_columns())
