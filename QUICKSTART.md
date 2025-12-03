# Quick Start Guide

## 3-Step Setup

### Step 1: Install Dependencies

```bash
pip install pandas numpy scipy pyyaml
```

### Step 2: Run Analysis

```bash
python main.py
```

### Step 3: Check Results

```bash
ls outputs/run_*/reports/
```

## Expected Output

```
outputs/run_YYYYMMDD_HHMMSS/
├── logs/
│   ├── orchestrator.log
│   ├── dataagent.log
│   ├── insightagent.log
│   ├── evaluator.log
│   └── creativegenerator.log
└── reports/
    ├── insights.json
    ├── recommendations.json
    └── report.md
```

## Configuration

Edit `config.yaml` to customize:

```yaml
analysis:
  baseline_days: 14  # Historical period
  current_days: 7    # Recent period

thresholds:
  roas_drop_severe: 0.30
  confidence_minimum: 0.60
```

## Custom CSV

```bash
python main.py --csv path/to/your/data.csv
```

## Testing

```bash
python test_evaluator.py
```

## Need Help?

1. Check `README.md` for full documentation
2. Review logs in `outputs/run_*/logs/`
3. See `ARCHITECTURE.md` for system design

## Key Features

✅ Schema validation before analysis  
✅ Baseline vs current comparison  
✅ Statistical tests (t-test, Mann-Whitney, Cohen's d)  
✅ Evidence-based insights  
✅ Confidence scoring (0.0 - 1.0)  
✅ Creative recommendations linked to issues  
✅ Comprehensive error handling  
✅ Per-agent logging  
✅ Timestamped outputs  

## Production-Grade Features

- **No silent failures**: All errors logged with context
- **Config-driven**: No magic numbers in code
- **Observable**: Decision logs explain "why"
- **Testable**: Unit tests included
- **Resilient**: Handles missing data, NaNs, empty groups

## File Structure

```
Main files:
├── main.py              # Entry point
├── orchestrator.py      # Pipeline coordinator
├── data_agent.py        # Data loading & validation
├── insight_agent.py     # Hypothesis generation
├── evaluator.py         # Statistical validation
├── creative_generator.py # Creative recommendations
├── schema.py            # Data schema & validation
├── utils.py             # Logging, config, helpers
└── config.yaml          # Configuration

Documentation:
├── README.md            # Full documentation
├── ARCHITECTURE.md      # System design
├── QUICKSTART.md        # This file

Testing:
├── test_evaluator.py    # Unit tests
├── requirements.txt     # Dependencies
└── Makefile            # Build automation
```

## What Makes This V2?

| Feature | V1 | V2 Production |
|---------|-----|---------------|
| Insights | Generic | Evidence-based |
| Validation | If/else | Statistical + confidence |
| Evidence | Basic | Baseline vs current + deltas |
| Error Handling | Try/except | Comprehensive with context |
| Schema | Implicit | Explicit with validation |
| Logging | Basic | Per-agent + decision logs |
| Testing | None | Unit tests |
| Config | Hardcoded | Config-driven |

**V2 proves you can own a production system end-to-end.**
