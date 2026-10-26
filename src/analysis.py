"""Bottlenecks, cause tests and a what-if simulation.

All numbers come from this file, not from a language model. The tests report
ASSOCIATIONS (cases with X take longer) and say so; they do not prove a cause."""
import numpy as np
import pandas as pd
from scipy import stats
from process_stats import case_table, transitions, rework

MIN_GROUP = 30   # below this many cases in a group we refuse to test


def bottlenecks(df, top=5):
    """Which hand-offs account for the most waiting. Shares add up to 100% of case time."""
    t = transitions(df)
    total = t["total_wait_h"].sum()
    t["share_of_elapsed"] = t["total_wait_h"] / total
    return t.head(top).reset_index(drop=True)


def compare_durations(cases, mask, label, seed=0, n_boot=2000):
    """Do cases where `mask` is True take longer than the rest? Rank test, effect size, bootstrap interval."""
    mask = pd.Series(mask, index=cases.index).astype(bool)
    a, b = cases.loc[mask, "duration_days"].to_numpy(), cases.loc[~mask, "duration_days"].to_numpy()
    out = {"metric_id": f"compare:{label}", "n_with": int(len(a)), "n_without": int(len(b))}
    if min(len(a), len(b)) < MIN_GROUP:
        return {**out, "enough_data": False}
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    rng = np.random.default_rng(seed)
    diffs = [np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b))) for _ in range(n_boot)]
    return {**out, "enough_data": True,
            "median_with_days": float(np.median(a)), "median_without_days": float(np.median(b)),
            "median_diff_days": float(np.median(a) - np.median(b)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_value": float(u.pvalue), "cliffs_delta": float(2 * u.statistic / (len(a) * len(b)) - 1),
            "note": "association, not proof of cause"}


def attribute_effects(df, cases, attr, min_group=MIN_GROUP):
    """Compare each category of a case attribute against all other cases. Bonferroni adjusted for the number of tests."""
    per_case = df.groupby("case_id")[attr].first()
    c = cases.set_index("case_id").join(per_case).reset_index()
    cats = [k for k, n in c[attr].value_counts().items() if n >= min_group and pd.notna(k)]
    res = [compare_durations(c, c[attr] == k, f"{attr}={k}") for k in cats]
    for r in res:
        if r.get("enough_data"):
            r["p_adjusted"] = min(1.0, r["p_value"] * len(res))
    return sorted([r for r in res if r.get("enough_data")], key=lambda r: r["p_value"])


def loop_span_days(df, activity):
    """For cases where `activity` happens more than once: time from its first to its last occurrence."""
    d = df[df["activity"] == activity].groupby("case_id")["timestamp"].agg(["min", "max", "size"])
    span = ((d["max"] - d["min"]).dt.total_seconds() / 86400).where(d["size"] > 1, 0.0)
    return span


def simulate_fix(cases, span, effectiveness=(0.3, 0.7), n_sims=5000, seed=0, resample=True):
    """What if a fix removed a share of the rework loop time? Effectiveness is unknown, so draw it from a range.
    Optimistic by construction: it assumes removed loop time is not moved to another step and no valid cases get rejected."""
    rng = np.random.default_rng(seed)
    s = span.reindex(cases["case_id"]).fillna(0.0).to_numpy()
    dur = cases["duration_days"].to_numpy()
    n = len(s)
    saved, rel = np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        idx = rng.integers(0, n, n) if resample else np.arange(n)
        eff = rng.uniform(*effectiveness)
        saved[i] = (eff * s[idx]).mean()
        rel[i] = saved[i] / dur[idx].mean()
    return {"metric_id": "simulation:remove_rework", "affected_share": float((s > 0).mean()),
            "mean_saving_days_median": float(np.median(saved)),
            "mean_saving_days_p5": float(np.percentile(saved, 5)), "mean_saving_days_p95": float(np.percentile(saved, 95)),
            "relative_saving_median": float(np.median(rel)), "effectiveness_range": list(effectiveness),
            "assumptions": "removed loop time is not shifted to another step; no valid cases wrongly rejected"}


def build_findings(df, top_bottlenecks=3, attrs=()):
    """Everything the agent may talk about, each with an id and example case ids as evidence."""
    cases = case_table(df)
    findings = []
    b = bottlenecks(df, top_bottlenecks)
    for i, r in b.iterrows():
        findings.append({"finding_id": r["metric_id"], "kind": "bottleneck",
                         "numbers": {k: (float(r[k]) if k != "count" else int(r[k])) for k in
                                     ("count", "median_wait_h", "p90_wait_h", "share_of_elapsed")},
                         "step": f"{r['activity']} > {r['next_activity']}"})
    rw = rework(df)
    for _, r in rw.head(2).iterrows():
        act = r["activity"]
        span = loop_span_days(df, act)
        has = cases["case_id"].isin(span[span > 0].index)
        findings.append({"finding_id": r["metric_id"], "kind": "rework",
                         "numbers": {"cases": int(r["cases"]), "share_of_cases": float(r["share_of_cases"])},
                         "evidence_cases": sorted(span[span > 0].index)[:5]})
        findings.append({"finding_id": f"compare:rework_{act}", "kind": "association",
                         "numbers": compare_durations(cases, has, f"rework_{act}")})
        findings.append({"finding_id": "simulation:remove_rework", "kind": "simulation",
                         "numbers": simulate_fix(cases, span)}) if act == rw.iloc[0]["activity"] else None
    for a in attrs:
        for r in attribute_effects(df, cases, a)[:3]:
            findings.append({"finding_id": r["metric_id"], "kind": "association", "numbers": r})
    return findings
