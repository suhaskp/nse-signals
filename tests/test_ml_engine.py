import numpy as np
import pandas as pd
import pytest

from models.ml_engine import (FEATURE_COLUMNS, MomentumClassifier, PurgedWalkForwardSplit,
                              build_features, build_inference_row, build_training_dataset,
                              train_and_evaluate)
from tests.conftest import make_ohlcv


def test_features_have_no_lookahead(ohlcv, ind_cfg):
    """Changing bar t's High/Low/Close/Volume must not change features at <= t."""
    base = build_features(ohlcv, ind_cfg)
    t = 300
    tampered = ohlcv.copy()
    tampered.iloc[t:, tampered.columns.get_indexer(["High", "Low", "Close", "Volume"])] *= 3
    after = build_features(tampered, ind_cfg)
    pd.testing.assert_frame_equal(base.iloc[: t + 1], after.iloc[: t + 1])


def test_inference_row_matches_training_features(ohlcv, ind_cfg):
    hist, today = ohlcv.iloc[:-1], ohlcv.iloc[-1]
    row = build_inference_row(hist, today["Open"], ohlcv.index[-1].date(), ind_cfg)
    expected = build_features(ohlcv, ind_cfg).iloc[[-1]]
    pd.testing.assert_frame_equal(row, expected, check_freq=False)


def test_inference_rejects_future_history(ohlcv, ind_cfg):
    with pytest.raises(ValueError):
        build_inference_row(ohlcv, 100.0, ohlcv.index[-1].date(), ind_cfg)


def test_purged_split_is_ordered_and_embargoed():
    dates = pd.Series(np.repeat(pd.bdate_range("2024-01-01", periods=300), 3))
    for tr, te in PurgedWalkForwardSplit(n_splits=4, embargo=5, min_train_dates=100).split(dates):
        train_d, test_d = dates.iloc[tr], dates.iloc[te]
        assert train_d.max() < test_d.min()
        gap = len(pd.bdate_range(train_d.max(), test_d.min())) - 2
        assert gap >= 5
        assert set(train_d).isdisjoint(set(test_d))


def test_train_predict_save_load(ind_cfg, model_cfg):
    histories = {f"T{i}": make_ohlcv(n=350, seed=i) for i in range(4)}
    X, y, meta = build_training_dataset(histories, ind_cfg)
    assert list(X.columns) == FEATURE_COLUMNS and meta["date"].is_monotonic_increasing
    model, report = train_and_evaluate(histories, ind_cfg, model_cfg)
    assert report["folds"] and 0 <= report["summary"]["mean_auc"] <= 1
    p = model.predict_proba(X.head(10))
    assert p.shape == (10,) and ((p >= 0) & (p <= 1)).all()
    loaded = MomentumClassifier.load(model_cfg, model.save())
    np.testing.assert_allclose(loaded.predict_proba(X.head(10)), p)
    assert not loaded.is_stale(7)
