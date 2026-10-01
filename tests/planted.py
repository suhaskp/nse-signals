import numpy as np, pandas as pd
def planted_market(n_stocks=60, n_days=1300, seed=3, strength=0.0012):
    """Stocks whose slowly-drifting trend persists: momentum genuinely predicts returns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-09-25", periods=n_days)
    out = {}
    for i in range(n_stocks):
        drift = np.zeros(n_days)
        for t in range(1, n_days):
            drift[t] = 0.995 * drift[t - 1] + rng.normal(0, strength * 0.1)
        r = drift + rng.normal(0, 0.015, n_days)
        c = 500 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * (1 + rng.normal(0, 0.002, n_days))
        out[f"P{i:02d}"] = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.005, "Low": np.minimum(o, c) * 0.995,
                                         "Close": c, "Volume": np.full(n_days, 2e6)}, index=idx)
    bc = 20000 * np.exp(np.cumsum(rng.normal(0.0003, 0.009, n_days)))
    bench = pd.DataFrame({"Open": bc, "High": bc * 1.004, "Low": bc * 0.996, "Close": bc, "Volume": 1e6}, index=idx)
    return out, bench
