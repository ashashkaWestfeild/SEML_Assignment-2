"""TEST TYPE 3 - ML BEHAVIOURAL TESTS: MODEL TRAINING (Objective 2.7a).

Ordinary unit tests cannot tell you whether *learning* actually happened. The
canonical ML training tests are implemented here:

  * overfit-a-small-batch  -- if the model cannot memorise 40 rows, the wiring
                              between features, labels and estimator is broken;
  * loss-decreases         -- log-loss must fall as capacity/iterations grow;
  * reproducibility        -- a fixed seed must give a bit-identical model;
  * boundary: single-class -- degenerate labels must raise, not silently fit;
  * boundary: tiny dataset -- model must still train on minimal viable rows.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import log_loss

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
    gradient-descent model.
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


def test_training_rejects_single_class_labels(xy):
    """Degenerate label distribution (single class) must raise
    ModelTrainingError rather than silently fitting a useless model.
    """
    features, labels = xy
    all_zeros = labels.head(100) * 0  # all labels = 0
    with pytest.raises(ModelTrainingError, match="single class"):
        ModelTrainer(settings).train(features.head(100), all_zeros, evaluate=False)


@pytest.mark.parametrize("n_rows", [10, 20, 50])
def test_training_succeeds_on_minimal_viable_datasets(xy, n_rows):
    """Model must successfully train even on very small datasets
    as long as both classes are present.
    """
    features, labels = xy
    small_x = features.head(n_rows)
    small_y = labels.head(n_rows)

    # Ensure both classes present for this slice
    if small_y.nunique() < 2:
        pytest.skip("Slice does not contain both classes")

    trainer = ModelTrainer(settings)
    trainer.train(small_x, small_y, evaluate=False)
    assert trainer.pipeline is not None
    proba = trainer.pipeline.predict_proba(small_x)
    assert proba.shape == (n_rows, 2)
    assert (proba >= 0).all() and (proba <= 1).all()
