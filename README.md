# SEML Assignment II — Loan Approval Risk Service (Group 84)

**BITS Pilani WILP · M.Tech AIML · AIMLCZG546 — Software Engineering for Machine Learning**

A production-grade refactor of the Assignment I loan-underwriting prototype,
extended with a REST API and a four-layer test suite. The system scores
consumer-credit applications in real time and returns an approval decision, a
calibrated probability and a risk tier.

---

## Submission files

| File | What it is |
|---|---|
| **[Group_84.pdf](Group_84.pdf)** | The report — 21 pages covering all nine tasks with figures, code listings and measured results |
| **[Group_84.ipynb](Group_84.ipynb)** | Executed notebook — every output is a real execution result, not a transcript |
| [Group_84/](Group_84/) | The source tree ([details](Group_84/README.md)) |
| [Assignment II.pdf](Assignment%20II.pdf) | The question paper |

---

## Measured results

| | Gate | Measured |
|---|---|---|
| Accuracy | ≥ 0.80 | **0.9480** |
| F1 | ≥ 0.80 | **0.9343** |
| ROC-AUC | ≥ 0.85 | **0.9881** |
| Brier score (calibration) | ≤ 0.15 | **0.0458** |
| Mean inference latency | ≤ 150 ms | **10.3 ms** |
| Schema conformance | = 1.00 | **1.0000** |
| Missing-value fraction | ≤ 0.02 | **0.0000** |

**Tests:** 89 passing across 4 types (unit · integration · data validation · ML behavioural), 93% line coverage.
**Code quality:** flake8 59 → **0** violations · black/isort clean · pylint **10.00/10**.

---

## What the tests caught

The QA suite is not decorative — it found real defects before submission:

* **Non-deterministic scoring.** An invariance test failed because `n_jobs=-1`
  made the forest accumulate tree votes in thread-completion order. The spread
  was ~5×10⁻¹⁷ — invisible in any aggregate metric, and enough to flip an
  applicant sitting exactly on the 0.50 cut-off between APPROVED and DENIED
  across identical requests. Unacceptable in regulated lending; the estimator
  is now pinned to a deterministic reduction.
* **A vacuous test.** `test_quality_gates_reject_a_weak_model` guarded its
  assertion behind an `if`, so it could report success without exercising the
  gate. Now unconditional.
* **An under-tested critical path.** Coverage showed `data/ingestion.py` at
  64% — the module the report singles out for its error handling was the
  least-exercised in the codebase. Now 100%.

---

## Quick start

```bash
pip install -r Group_84/requirements.txt
```

```bash
cd Group_84 && python data/generate_synthetic_data.py --rows 20000
```

```bash
python data/prepare_data.py && python scripts/train_model.py
```

```bash
python -m pytest tests -v
```

```bash
PYTHONPATH=src python -m uvicorn loan_risk.api.app:app --port 8000
```

Then open <http://127.0.0.1:8000/docs> for the generated Swagger UI.

Full structure, module map and tooling commands: **[Group_84/README.md](Group_84/README.md)**.

---

## Group 84

| Sl. | BITS ID | Name |
|:--:|:--|:--|
| 1 | 2025AA05710 | Singh Pritesh |
| 2 | 2025AA05368 | Gangera Tushar |
| 3 | 2025AB05154 | Gangam Shuba Nandini |
| 4 | 2025AA05574 | Shaifali Garg |

Previous assignment: [SEML_Assignment-1](https://github.com/ashashkaWestfeild/SEML_Assignment-1)
