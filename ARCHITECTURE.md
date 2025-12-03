# Architecture Documentation

## System Overview

Kasparro V2 Production is a data-driven Facebook Ads performance analysis system built with production-grade engineering practices.

## Design Principles

### 1. Evidence-Based Decision Making
- Every insight must be backed by concrete evidence
- Baseline vs current comparison for all metrics
- Statistical validation with confidence scores

### 2. Resilience First
- No silent failures
- Comprehensive error handling
- Graceful degradation

### 3. Observability
- Structured logging at every step
- Decision logs explaining "why"
- Timestamped run directories

### 4. Config-Driven
- No magic numbers in code
- All thresholds in config.yaml
- Easy tuning without code changes

## Component Architecture

### Schema Layer (`schema.py`)

**Purpose**: Data contract enforcement

**Key Classes**:
- `ColumnSchema`: Single column definition with validation rules
- `DataSchema`: Complete schema with all columns
- `validate_and_clean_data()`: Main validation function

**Validation Rules**:
- Required columns
- Data types (numeric, string, date, categorical)
- Nullable constraints
- Range constraints (min/max)
- Allowed values for categoricals

**Error Handling**:
```python
validation_result = {
    "valid": True/False,
    "errors": ["Error 1", "Error 2"],
    "warnings": ["Warning 1"],
    "row_count": 1000,
    "columns_validated": ["col1", "col2"]
}
```

### Utilities Layer (`utils.py`)

**Purpose**: Cross-cutting concerns

**Key Components**:

1. **ProductionLogger**
   - Structured JSON logging
   - Per-agent log files
   - Decision logging
   - Input/output summaries

2. **ConfigLoader**
   - YAML configuration loading
   - Config validation
   - Required section checks

3. **ErrorHandler**
   - Centralized error handling
   - Context-aware error messages
   - Missing column handling
   - Empty group handling
   - NaN/infinity handling

4. **Helper Functions**
   - `safe_divide()`: Zero-division protection
   - `safe_percentage_change()`: NaN-safe percentage calculation
   - `create_run_directory()`: Timestamped output directories

### Data Agent (`data_agent.py`)

**Purpose**: Data loading, validation, and baseline computation

**Key Methods**:

1. `load_and_validate(csv_path)`
   - Reads CSV
   - Validates schema
   - Cleans data
   - Returns success/failure with errors

2. `compute_baseline_and_current()`
   - Splits data into baseline and current periods
   - Validates minimum sample sizes
   - Returns both datasets with metadata

3. `analyze_segments(data, dimensions)`
   - Groups by dimensions
   - Aggregates metrics
   - Handles empty groups
   - Returns segment analysis

**Error Handling**:
- File not found
- CSV parse errors
- Schema validation failures
- Missing columns
- Empty datasets

### Insight Agent (`insight_agent.py`)

**Purpose**: Evidence-based hypothesis generation

**Key Methods**:

1. `generate_insights(baseline_data, current_data)`
   - Compares baseline to current
   - Identifies performance changes
   - Generates hypotheses with evidence
   - Returns prioritized insights

2. `_analyze_overall_performance()`
   - ROAS changes
   - CTR changes
   - Conversion rate changes

3. `_analyze_segments()`
   - Segment-level performance
   - Cross-dimensional analysis

4. `_detect_creative_fatigue()`
   - CTR trends by creative type
   - Frequency analysis
   - Fatigue indicators

5. `_detect_audience_shifts()`
   - Audience type performance
   - Targeting changes

**Insight Structure**:
```python
{
    "hypothesis": "Specific statement about what changed",
    "driver": "Root cause",
    "evidence_type": "creative_fatigue|segment_drop|audience_shift|metric_drop",
    "affected_metrics": ["roas", "ctr"],
    "affected_segments": {"creative_type": "Video"},
    "priority": "high|medium|low"
}
```

### Evaluator (`evaluator.py`)

**Purpose**: Statistical validation with evidence

**Key Methods**:

1. `evaluate_hypothesis(hypothesis, baseline, current, metrics, segments)`
   - Applies filters
   - Analyzes each metric
   - Runs statistical tests
   - Computes confidence
   - Returns ValidationResult

2. Statistical Tests:
   - **T-test**: Parametric test for mean differences
   - **Mann-Whitney U**: Non-parametric alternative
   - **Cohen's d**: Effect size measurement

3. `_assess_impact()`: High/medium/low based on metric drops

4. `_assess_severity()`: Severe/moderate/minor based on drop magnitude

5. `_compute_confidence()`: 0.0-1.0 based on:
   - Statistical significance
   - Effect size
   - Sample size
   - Consistency across metrics

**Evidence Structure**:
```python
{
    "metric": "roas",
    "baseline_value": 3.0,
    "current_value": 2.0,
    "absolute_delta": -1.0,
    "relative_delta_pct": -33.3,
    "segment": "creative_type",
    "segment_value": "Video"
}
```

**Validation Result**:
```python
{
    "hypothesis": "...",
    "validated": True/False,
    "impact": "high|medium|low",
    "confidence": 0.85,
    "evidence": [...],
    "statistical_tests": {...},
    "severity": "severe|moderate|minor",
    "recommendation": "..."
}
```

### Creative Generator (`creative_generator.py`)

**Purpose**: Issue-specific creative recommendations

**Key Methods**:

1. `generate_recommendations(insights, validations)`
   - Filters to validated insights only
   - Generates recommendations per insight
   - Links to specific evidence
   - Returns creative recommendations

2. `_recommendations_for_fatigue()`
   - New creative types
   - Fresh messaging angles
   - Testing strategies

3. `_recommendations_for_segment_drop()`
   - Segment-specific creatives
   - Targeted messaging

4. `_recommendations_for_audience_shift()`
   - Audience diversification
   - Complementary targeting

5. `_recommendations_for_metric_drop()`
   - CTR recovery tactics
   - ROAS optimization

**Creative Recommendation Structure**:
```python
{
    "title": "Action-oriented title",
    "creative_type": "Image|Video|Carousel|UGC",
    "messaging_angle": "Specific messaging approach",
    "target_segment": {"dimension": "value"},
    "rationale": "Why this recommendation (with evidence)",
    "linked_insight": "The hypothesis this addresses",
    "expected_impact": "high|medium|low",
    "testing_strategy": "How to test this",
    "success_metrics": ["ctr", "roas"]
}
```

### Orchestrator (`orchestrator.py`)

**Purpose**: Pipeline coordination and output generation

**Pipeline Steps**:

1. **Initialize**
   - Load config
   - Create run directory
   - Setup logging
   - Initialize all agents

2. **Load & Validate Data**
   - Call DataAgent.load_and_validate()
   - Check for errors
   - Fail fast if invalid

3. **Compute Periods**
   - Call DataAgent.compute_baseline_and_current()
   - Validate sample sizes
   - Log metadata

4. **Generate Insights**
   - Call InsightAgent.generate_insights()
   - Log insights generated

5. **Validate Insights**
   - For each insight:
     - Call Evaluator.evaluate_hypothesis()
     - Log validation result
   - Count validated insights

6. **Generate Creatives**
   - Call CreativeGenerator.generate_recommendations()
   - Log recommendations

7. **Generate Outputs**
   - insights.json
   - recommendations.json
   - report.md
   - All logs

**Error Handling**:
- Try/except at each step
- Log all errors with context
- Continue or fail based on config
- Return structured result

## Data Flow Diagram

```
┌────────────────────┐
│   CSV File Input   │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Schema Validation  │◄─── schema.py
└─────────┬──────────┘
          │ (valid)
          ▼
┌────────────────────┐
│   Data Loading     │◄─── data_agent.py
│   & Cleaning       │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Baseline vs        │
│ Current Split      │
└─────┬─────┬────────┘
      │     │
      │     └──────────┐
      │                │
      ▼                ▼
┌──────────┐    ┌──────────┐
│ Baseline │    │ Current  │
│ (14 days)│    │ (7 days) │
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Insight Agent  │◄─── insight_agent.py
    │ (Generate      │
    │  Hypotheses)   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Evaluator    │◄─── evaluator.py
    │ (Validate with │
    │  Statistics)   │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │   Creative     │◄─── creative_generator.py
    │   Generator    │
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │ Output Files   │
    │ • insights.json│
    │ • recs.json    │
    │ • report.md    │
    └────────────────┘
```

## Error Handling Strategy

### Levels of Error Handling

1. **Component Level**
   - Try/except in each method
   - Log error with context
   - Return error result (don't crash)

2. **Agent Level**
   - Validate inputs
   - Handle missing data
   - Provide fallback values

3. **Orchestrator Level**
   - Check agent results
   - Decide: continue or fail
   - Aggregate errors

4. **System Level**
   - Catch unhandled exceptions
   - Log to system log
   - Return error code

### Error Categories

1. **Fatal Errors** (stop pipeline):
   - File not found
   - Schema validation failure (strict mode)
   - Config loading failure

2. **Recoverable Errors** (log and continue):
   - Empty segment
   - Missing optional column
   - Single metric failure

3. **Warnings** (log only):
   - Low sample size
   - High NaN rate
   - Extra columns

## Configuration Strategy

### Config-Driven Design

All tunable parameters in `config.yaml`:

```yaml
analysis:
  baseline_days: 14      # Easy to change
  current_days: 7
  min_sample_size: 30

thresholds:
  roas_drop_severe: 0.30  # No magic numbers in code
  p_value_threshold: 0.05
```

### Why Config-Driven?

1. **No code changes** for tuning
2. **Different environments** (dev, staging, prod)
3. **A/B testing** thresholds
4. **Documentation** - config is self-documenting

## Logging Strategy

### Log Levels

- **DEBUG**: Detailed diagnostic info
- **INFO**: Major steps and decisions
- **WARNING**: Potential issues
- **ERROR**: Failures with context

### Log Structure (JSON)

```json
{
  "timestamp": "2025-01-02T14:30:22",
  "name": "DataAgent",
  "level": "INFO",
  "message": "Data loaded successfully",
  "rows": 1000,
  "columns": 15
}
```

### Decision Logs

```json
{
  "decision": "Baseline and current periods computed",
  "reason": "Baseline: 14 days, Current: 7 days",
  "data": {
    "baseline_rows": 420,
    "current_rows": 210
  }
}
```

## Testing Strategy

### Unit Tests

- **Component-level tests**: Each agent
- **Edge cases**: Empty data, NaNs, missing columns
- **Error handling**: Verify graceful failures

### Integration Tests (Future)

- **Full pipeline**: CSV to outputs
- **Multiple datasets**: Different sizes, schemas
- **Config variations**: Different thresholds

## Performance Considerations

### Current Performance

- 10K rows: ~15 seconds
- 100K rows: ~90 seconds

### Optimization Opportunities

1. **Parallel validation**: Validate insights concurrently
2. **Caching**: Cache aggregations
3. **Sampling**: Sample large datasets for insights
4. **Database**: Move from CSV to database

## Extension Points

### Adding New Insight Types

1. Edit `insight_agent.py`
2. Add new `_detect_*()` method
3. Return `Insight` objects
4. No other changes needed

### Adding New Statistical Tests

1. Edit `evaluator.py`
2. Add test in `_analyze_metric()`
3. Add to `statistical_tests` dict
4. Include in confidence calculation

### Adding New Output Formats

1. Edit `orchestrator.py`
2. Add new `_generate_*()` method
3. Call in `_generate_outputs()`

## Security Considerations

### Current Implementation

- **CSV injection**: Not validated (add for production)
- **Path traversal**: Not validated (add for production)
- **Config injection**: YAML safe load (✓)

### Production Additions Needed

1. Input sanitization
2. File upload validation
3. Rate limiting
4. Authentication/authorization
5. Audit logging

## Deployment Considerations

### Current State (Local)

- CLI tool
- Local file system
- No authentication

### Production Additions Needed

1. **API**: REST or GraphQL endpoints
2. **Database**: PostgreSQL for data, Redis for cache
3. **Queue**: Celery for async processing
4. **Monitoring**: Prometheus + Grafana
5. **Alerting**: Slack/email notifications
6. **CI/CD**: GitHub Actions + Docker

---

**This architecture is designed for:**
- ✅ Clarity over cleverness
- ✅ Resilience over optimization
- ✅ Observability over abstraction
- ✅ Production-readiness over quick hacks
