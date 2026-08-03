# Loan Approval Risk Service — Assignment II (Group 84)

**BITS Pilani WILP · M.Tech AIML · AIMLCZG546 Software Engineering for Machine Learning**

Production-grade refactor of the Assignment I loan-underwriting system, extended
with a REST API, a four-layer test suite, and measured model- and data-quality
metrics.

| Deliverable | Where |
|---|---|
| Report (submission) | `../Group_84.pdf` |
| Executed notebook (submission) | `../Group_84.ipynb` |
| Research-code exemplar | `notebooks/research_prototype.ipynb`, `legacy/research_feature_prototype.py` |
| Production package | `src/loan_risk/` |
| Test suite | `tests/` (84 tests) |
| Lint evidence | `reports/lint/` |
| Metrics & figures | `reports/metrics/`, `reports/figures/` |

---

## Project structure

```
Group_84/
├── configs/config.yaml            # every threshold, gate and hyper-parameter
├── data/
│   ├── generate_synthetic_data.py # schema-faithful stand-in for the Kaggle file
│   ├── prepare_data.py            # cleaning, encoding, feature engineering
│   └── loan_data_processed.csv    # 20,000 x 23 modelling table
├── src/loan_risk/                 # THE PRODUCTION PACKAGE
│   ├── config.py                  # frozen dataclasses loaded from YAML
│   ├── exceptions.py              # domain exception hierarchy
│   ├── logging_utils.py           # structured JSON logging + Timer
│   ├── data/
│   │   ├── ingestion.py           # DataIngestor      (OOP)
│   │   └── validation.py          # DataValidator     (schema contract, DQ-1/DQ-2)
│   ├── features/engineering.py    # FeatureEngineer   (sklearn transformer)
│   ├── models/
│   │   ├── trainer.py             # ModelTrainer      (train/evaluate/gate/persist)
│   │   └── predictor.py           # ModelRegistry + RiskPredictor (pipe-and-filter)
│   ├── monitoring/drift.py        # DriftMonitor      (PSI, KS -> DQ-3/DQ-4)
│   └── api/
│       ├── schemas.py             # Pydantic request/response contract
│       └── app.py                 # FastAPI routes + error mapping
├── tests/                         # unit | integration | data | ml
├── scripts/                       # train, evaluate, render evidence
├── legacy/                        # research code, preserved unformatted
├── reports/                       # lint reports, metrics JSON, figures
└── artifacts/model.joblib         # serialised pipeline (feature step included)
```

---

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python data/generate_synthetic_data.py --rows 20000
```

```bash
python data/prepare_data.py
```

```bash
python scripts/train_model.py
```

```bash
python -m pytest tests -v
```

```bash
PYTHONPATH=src python -m uvicorn loan_risk.api.app:app --port 8000
```

Then open <http://127.0.0.1:8000/docs> for the generated Swagger UI.

---

## Quality gates

The build fails if any of these regress (`configs/config.yaml → quality_gates`):

| Gate | Threshold | Measured |
|---|---|---|
| Accuracy | ≥ 0.80 | **0.9480** |
| F1 | ≥ 0.80 | **0.9343** |
| ROC-AUC | ≥ 0.85 | **0.9881** |
| Brier score | ≤ 0.15 | **0.0458** |
| Mean latency | ≤ 150 ms | **10.3 ms** |
| Missing-value fraction | ≤ 0.02 | **0.0000** |
| PSI (drift) | ≤ 0.20 | monitored per batch |

## Code quality

```bash
python -m isort src scripts tests
```

```bash
python -m black src scripts tests
```

```bash
python -m flake8 src scripts tests
```

```bash
python -m pylint src/loan_risk
```

Current state: **flake8 0 violations**, **black/isort clean**, **pylint 10.00/10**.
The `legacy/` directory is excluded from the formatters on purpose — it is the
"before" evidence for Objective 1.4.

---

## Group 84

| Sl. | BITS ID | Name |
|:--:|:--|:--|
| 1 | 2025AA05710 | Singh Pritesh |
| 2 | 2025AA05368 | Gangera Tushar |
| 3 | 2025AB05154 | Gangam Shuba Nandini |
| 4 | 2025AA05574 | Shaifali Garg |
