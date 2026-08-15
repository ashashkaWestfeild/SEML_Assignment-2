"""TEST TYPE 3 - ML BEHAVIOURAL TESTS: MODEL TRAINING (Objective 2.7a).

Ordinary unit tests cannot tell you whether *learning* actually happened. The
canonical ML training tests are implemented here:

  * overfit-a-small-batch  -- if the model cannot memorise 40 rows, the wiring
                              between features, labels and estimator is broken;
  * loss-decreases         -- log-loss must fall as capacity/iterations grow;
  * reproducibility        -- a fixed seed must give a bit-identical model;
  * label-shuffle sanity   -- with permuted labels, held-out AUC must collapse
                              to chance, proving the signal is real and not a
                              leak or an evaluation bug;
  * quality gates          -- the release gates themselves are tested.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from loan_risk.config import settings
from loan_risk.exceptions import ModelTrainingError
from loan_risk.models.trainer import ModelTrainer

pytestmark = pytest.mark.ml


def test_model_can_overfit_a_small_batch(xy):
    """A high-capacity model must memorise 40 rows (accuracy >= 0.95).

    Failure here means labels are misaligned with features, or the pipeline is
    dropping the signal before it reaches the estimator.
    """
    features, labels = xy
    small_x = features.head(40)
    small_y = labels.head(40)

    trainer = ModelTrainer(settings)
    trainer.train(small_x, small_y, evaluate=False)
    train_accuracy = trainer.pipeline.score(small_x, small_y)
    assert train_accuracy >= 0.95, f"Could not overfit 40 rows: {train_accuracy:.3f}"


def test_training_loss_decreases_with_capacity(xy):
    """Log-loss on the training batch must fall monotonically as depth grows.

    This is the tree-ensemble analogue of "the loss curve goes down" for a
    gradient-descent model. Depth -- not the number of trees -- is the right
    capacity axis here: an unconstrained forest already interpolates the batch
    at n_estimators=1, so sweeping tree count would measure variance, not
    learning.
    """
    features, labels = xy
    batch_x, batch_y = features.head(300), labels.head(300)

    losses = []
    for max_depth in (1, 3, 8, None):
        trainer = ModelTrainer(settings)
        pipeline = trainer.build_pipeline("random_forest")
        pipeline.set_params(
            model__n_estimators=60,
            model__max_depth=max_depth,
            model__min_samples_leaf=1,
        )
        pipeline.fit(batch_x, batch_y)
        probabilities = np.clip(pipeline.predict_proba(batch_x)[:, 1], 1e-9, 1 - 1e-9)
        losses.append(log_loss(batch_y, probabilities))

    assert losses == sorted(losses, reverse=True), f"Loss did not decrease: {losses}"
    assert losses[-1] < losses[0] / 2.0


def test_training_is_reproducible(xy):
    """Identical seed + identical data must give identical predictions."""
    features, labels = xy
    predictions = []
    for _ in range(2):
        trainer = ModelTrainer(settings)
        trainer.train(features, labels, evaluate=False)
        predictions.append(trainer.pipeline.predict_proba(features.head(50))[:, 1])
    np.testing.assert_allclose(predictions[0], predictions[1])


def test_shuffled_labels_destroy_generalisation(xy):
    """With permuted labels held-out AUC must fall back to ~0.5.

    If it does not, the evaluation split is leaking and every reported metric
    is meaningless.
    """
    features, labels = xy
    shuffled = labels.sample(frac=1.0, random_state=42).reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(
        features.reset_index(drop=True), shuffled, test_size=0.3, random_state=42
    )
    trainer = ModelTrainer(settings)
    pipeline = trainer.build_pipeline("random_forest")
    pipeline.fit(x_train, y_train)
    auc = roc_auc_score(y_test, pipeline.predict_proba(x_test)[:, 1])
    assert 0.35 < auc < 0.65, f"Leakage suspected: AUC on shuffled labels = {auc:.3f}"


def test_quality_gates_reject_a_weak_model(xy):
    """The gate mechanism itself must fail a deliberately under-fit model."""
    features, labels = xy
    trainer = ModelTrainer(settings)
    pipeline = trainer.build_pipeline("random_forest")
    pipeline.set_params(model__n_estimators=1, model__max_depth=1)
    trainer.pipeline = pipeline

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=1
    )
    pipeline.fit(x_train.head(20), y_train.head(20))
    weak_metrics = trainer.evaluate(x_test, y_test)

    # Seeded end to end, so the stump's AUC (~0.53) is deterministic and this
    # assertion is unconditional -- a guarded version could pass vacuously.
    assert weak_metrics.roc_auc < settings.gates.min_roc_auc
    with pytest.raises(ModelTrainingError, match="Quality gates breached"):
        trainer.enforce_quality_gates(weak_metrics)


@pytest.mark.parametrize(
    "scenario",
    ["single_class_target", "unknown_algorithm"],
)
def test_training_rejects_unusable_inputs(xy, scenario):
    """Each precondition failure surfaces as one ModelTrainingError."""
    features, labels = xy
    trainer = ModelTrainer(settings)
    with pytest.raises(ModelTrainingError):
        if scenario == "single_class_target":
            trainer.train(features, labels * 0, evaluate=False)
        elif scenario == "length_mismatch":
            trainer.train(features, labels.head(10), evaluate=False)
        else:
            trainer.build_pipeline("deep_neural_magic")


def test_model_quality_metrics_meet_release_gates(xy):
    """MODEL-QUALITY METRICS (Objective 2.8a): accuracy, F1, ROC-AUC, Brier."""
    features, labels = xy
    trainer = ModelTrainer(settings)
    trainer.train(features, labels)
    metrics = trainer.metrics

    assert metrics.accuracy >= settings.gates.min_accuracy
    assert metrics.f1 >= settings.gates.min_f1
    assert metrics.roc_auc >= settings.gates.min_roc_auc
    # Calibration: a Brier score this low means the probability itself is
    # trustworthy enough to drive risk tiering, not just the 0/1 decision.
    assert metrics.brier_score <= settings.gates.max_brier_score
    trainer.enforce_quality_gates()


def test_artifact_round_trips_through_disk(xy, tmp_path):
    """Persist -> reload must preserve predictions exactly."""
    import joblib

    features, labels = xy
    trainer = ModelTrainer(settings)
    trainer.train(features, labels)
    path = trainer.save(tmp_path / "model.joblib")

    bundle = joblib.load(path)
    reloaded = bundle["pipeline"]
    np.testing.assert_allclose(
        trainer.pipeline.predict_proba(features.head(20))[:, 1],
        reloaded.predict_proba(features.head(20))[:, 1],
    )
    assert bundle["features"] == settings.model.features
