import sys, pathlib
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from analysis import compare_durations, simulate_fix, bottlenecks, attribute_effects, build_findings
from load_log import normalise
from process_stats import case_table

def cases_with(effect, n=400, seed=1):
    rng = np.random.default_rng(seed)
    has = rng.random(n) < 0.3
    d = rng.gamma(4, 1.0, n) + np.where(has, effect, 0.0)
    return pd.DataFrame({"case_id": [f"c{i}" for i in range(n)], "duration_days": d}), has

def test_planted_delay_is_detected():
    cases, has = cases_with(2.0)
    r = compare_durations(cases, has, "x")
    assert r["p_value"] < 0.001 and r["ci_low"] > 0 and abs(r["median_diff_days"] - 2.0) < 0.7

def test_no_effect_is_not_flagged():
    cases, has = cases_with(0.0)
    r = compare_durations(cases, has, "x")
    assert r["p_value"] > 0.01 and r["ci_low"] < 0 < r["ci_high"]

def test_small_groups_are_refused():
    cases, _ = cases_with(2.0, n=100)
    assert compare_durations(cases, np.arange(100) < 10, "x")["enough_data"] is False

def test_simulation_exact_when_not_resampled():
    cases = pd.DataFrame({"case_id": ["a", "b"], "duration_days": [10.0, 10.0]})
    span = pd.Series({"a": 2.0, "b": 0.0})
    r = simulate_fix(cases, span, effectiveness=(0.5, 0.5), n_sims=50, resample=False)
    assert abs(r["mean_saving_days_median"] - 0.5) < 1e-9 and r["affected_share"] == 0.5

def test_simulation_interval_brackets_truth_with_resampling():
    cases = pd.DataFrame({"case_id": [f"c{i}" for i in range(200)], "duration_days": 10.0})
    span = pd.Series({f"c{i}": 2.0 for i in range(100)})
    r = simulate_fix(cases, span, effectiveness=(0.5, 0.5))
    assert r["mean_saving_days_p5"] < 0.5 < r["mean_saving_days_p95"]

def test_bottleneck_shares_sum_to_one():
    rows = [(c, a, t) for c in "AB" for a, t in (("S", "2024-01-01 00:00"), ("T", "2024-01-02 00:00"), ("U", "2024-01-05 00:00"))]
    df = normalise(pd.DataFrame([{"case:concept:name": c, "concept:name": a, "time:timestamp": t} for c, a, t in rows]))
    b = bottlenecks(df, 10)
    assert abs(b.share_of_elapsed.sum() - 1) < 1e-9 and b.iloc[0].activity == "T"

def test_multiple_testing_adjustment_and_null_attribute():
    rng = np.random.default_rng(3); n = 600
    rows = []
    for i in range(n):
        t = pd.Timestamp("2024-01-01") + pd.Timedelta(days=float(rng.gamma(4, 1)))
        rows += [{"case:concept:name": f"c{i}", "concept:name": "S", "time:timestamp": pd.Timestamp("2024-01-01"), "kind": "ABC"[i % 3]},
                 {"case:concept:name": f"c{i}", "concept:name": "E", "time:timestamp": t, "kind": "ABC"[i % 3]}]
    df = normalise(pd.DataFrame(rows)); cases = case_table(df)
    res = attribute_effects(df, cases, "kind")
    assert len(res) == 3 and all(r["p_adjusted"] >= r["p_value"] for r in res) and all(r["p_adjusted"] > 0.05 for r in res)
