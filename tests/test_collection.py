import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from collection import case_features, censoring_cutoff, analysis_set, fit_adjusted, delay_bands, scenario_faster_sending
from load_log import normalise

def synth(n=6000, true_or_per_30d=1.5, seed=0):
    """Fines with a PLANTED effect: each extra 30 days before sending multiplies the odds of collection."""
    rng = np.random.default_rng(seed)
    delay = rng.gamma(3, 30, n)                                   # days
    p = 1 / (1 + np.exp(-(-1.0 + np.log(true_or_per_30d) * delay / 30)))
    coll = rng.random(n) < p
    rows = []
    for i in range(n):
        t0 = pd.Timestamp("2005-01-01") + pd.Timedelta(days=int(rng.integers(0, 1000)))
        base = {"case:concept:name": f"c{i}", "amount": float(rng.integers(30, 400)), "article": float(rng.choice([157, 158, 142])),
                "vehicleClass": str(rng.choice(["A", "C", "M"])), "points": float(rng.integers(0, 4))}
        rows.append({**base, "concept:name": "Create Fine", "time:timestamp": t0})
        rows.append({**base, "concept:name": "Send Fine", "time:timestamp": t0 + pd.Timedelta(days=float(delay[i]))})
        if coll[i]:
            rows.append({**base, "concept:name": "Send for Credit Collection", "time:timestamp": t0 + pd.Timedelta(days=float(delay[i]) + 300)})
    return normalise(pd.DataFrame(rows))

def test_planted_effect_is_recovered():
    """A single random draw can miss a 95% interval 1 time in 20, so check 10 fixed seeds:
    the average estimate must be close to the truth and the interval must usually contain it."""
    est, hit = [], []
    for seed in range(10):
        d = analysis_set(case_features(synth(n=4000, seed=100 + seed)), cutoff=pd.Timestamp("2100-01-01"), max_delay_q=1.0)
        _, r = fit_adjusted(d)
        est.append(r["odds_ratio_per_30_days"]); hit.append(r["ci_low"] < 1.5 < r["ci_high"])
        assert r["p_value"] < 0.001
    assert abs(np.mean(est) - 1.5) < 0.06 and np.mean(hit) >= 0.8

def test_no_effect_is_not_flagged():
    df = synth(true_or_per_30d=1.0, seed=2)
    d = analysis_set(case_features(df), cutoff=pd.Timestamp("2100-01-01"), max_delay_q=1.0)
    d["start_year"] = d["created"].dt.year
    _, r = fit_adjusted(d)
    assert r["ci_low"] < 1.0 < r["ci_high"]

def test_censoring_cutoff_excludes_recent_cases():
    feat = pd.DataFrame({"to_collection_days": [100.0, 200.0, 300.0, 400.0]})
    cutoff, window = censoring_cutoff(feat, pd.Timestamp("2013-06-17"), q=1.0)
    assert window == 400.0 and cutoff == pd.Timestamp("2013-06-17") - pd.Timedelta(days=400)

def test_bands_cover_all_cases_and_rates_rise_with_planted_effect():
    d = analysis_set(case_features(synth()), cutoff=pd.Timestamp("2100-01-01"), max_delay_q=1.0)
    b = delay_bands(d)
    assert b["cases"].sum() == len(d) and b["collection_rate"].is_monotonic_increasing

def test_scenario_direction_and_zero_change():
    d = analysis_set(case_features(synth()), cutoff=pd.Timestamp("2100-01-01"), max_delay_q=1.0)
    d["start_year"] = d["created"].dt.year
    m, _ = fit_adjusted(d)
    s = scenario_faster_sending(d, m, days_faster=30, n_draws=100)
    assert s["change_pp_p95"] < 0                       # sending faster lowers the modelled collection rate
    z = scenario_faster_sending(d, m, days_faster=0, n_draws=50)
    assert abs(z["change_pp_median"]) < 1e-9

def test_leaky_columns_are_never_features():
    import collection
    for leak in ("dismissal", "expense", "notificationType", "totalPaymentAmount", "paymentAmount", "lastSent"):
        assert leak not in collection.FORMULA

def test_tiny_year_groups_are_dropped():
    d = analysis_set(case_features(synth(n=1000)), cutoff=pd.Timestamp("2100-01-01"), max_delay_q=1.0, min_year_cases=10**6)
    assert len(d) == 0

def test_non_convergence_is_an_error_not_a_result():
    import pytest
    from collection import _fit
    d = pd.DataFrame({"collected_flag": [0, 0, 0, 1, 1, 1] * 5, "send_delay_30d": [0, 0, 0, 1, 1, 1] * 5})
    with pytest.raises(Exception):
        _fit("collected_flag ~ send_delay_30d", d)         # perfectly separated data
