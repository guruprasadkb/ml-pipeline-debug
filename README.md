# ML Pipeline Debugging Challenge

## Scenario

You've inherited a fraud detection ML pipeline. The model was reported to achieve
**99% accuracy** during development, but performance dropped to **~50% in production**.

## Your Task

1. Explore the codebase and data
2. Identify all bugs (there are at least 4)
3. Fix each bug and explain why it matters
4. Run `pytest -v` to verify your fixes

## Project Structure

```
src/
  pipeline.py         # Main orchestration — start here
  data_loader.py      # Data loading and train/test splitting
  preprocessing.py    # Feature preprocessing / scaling
  model.py            # Model training and evaluation
  evaluation.py       # Metrics reporting
data/
  fraud_detection.csv # 1000 transactions (5% fraud rate)
tests/
  test_preprocessing.py
  test_model.py
notebooks/
  exploration.ipynb   # Starter notebook for data exploration
```

## Quick Start

```bash
# Run the pipeline
python -m src.pipeline

# Run tests (expect failures until bugs are fixed)
pytest -v
```

## Known Issues

The business team reported:
- Model accuracy drops dramatically on new data
- Predictions are biased toward the majority class
- Stakeholders don't trust the reported metrics
