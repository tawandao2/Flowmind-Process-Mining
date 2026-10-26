"""Turns the analysis results into a findings file. Every finding has an id and a list of
'facts': sentences with the numbers already formatted. The report writer may only use these."""
import json, sys
import numpy as np
import pandas as pd
from process_stats import case_table, variants, transitions
import collection as col


def pct(x): return f"{x * 100:.1f}%"


def build(df):
    cases = case_table(df)
    F = []
    add = lambda fid, kind, title, facts, must=False, numbers=None: F.append(
        {"id": fid, "kind": kind, "title": title, "facts": facts, "must_cover": must, "numbers": numbers or {}})

    add("overview", "descriptive", "The data", [
        f"{len(cases):,} traffic fines (cases), {len(df):,} events and {df['activity'].nunique()} different steps, "
        f"with fines created from {cases['start'].dt.year.min()} to {cases['start'].dt.year.max()}"])

    v = variants(df, 3)
    add("paths", "descriptive", "Most common paths", [
        f"{pct(r['share'])} of fines follow {r['variant']} with a median of {r['median_days']:.0f} days" for _, r in v.iterrows()], True)

    t = transitions(df)
    t["share"] = t["total_wait_h"] / t["total_wait_h"].sum()
    add("bottlenecks", "bottleneck", "Where the waiting is", [
        f"{r['activity']} to {r['next_activity']}: median wait {r['median_wait_h'] / 24:.0f} days, "
        f"{pct(r['share'])} of all elapsed time ({int(r['count']):,} occurrences)" for _, r in t.head(3).iterrows()])
    n = t[(t.activity == "Insert Fine Notification") & (t.next_activity == "Add penalty")].iloc[0]
    sub = df.sort_values(["case_id", "position"])
    sub["nxt"], sub["nt"] = sub.groupby("case_id")["activity"].shift(-1), sub.groupby("case_id")["timestamp"].shift(-1)
    w = sub[(sub.activity == "Insert Fine Notification") & (sub.nxt == "Add penalty")]
    hrs = (w.nt - w.timestamp).dt.total_seconds() / 3600
    add("fixed_wait", "descriptive", "A fixed waiting period", [
        f"{pct(((hrs - 1440).abs() <= 1).mean())} of waits from Insert Fine Notification to Add penalty are within 1 hour of exactly 60 days, "
        f"which looks like a built in rule and not a delay"])

    feat = col.case_features(df)
    cutoff, window = col.censoring_cutoff(feat, df["timestamp"].max())
    d = col.analysis_set(feat, cutoff)
    add("caveat_censoring", "caveat", "Unfinished cases", [
        f"Fines created after {cutoff.date()} are excluded because 99 percent of collections happen within {window:.0f} days, "
        f"and newer fines had not had time to finish; {len(d):,} sent fines from {d['start_year'].min()} to {d['start_year'].max()} are analysed"])
    b = col.delay_bands(d)
    add("bands", "association", "Collection rate by sending delay (raw)", [
        f"Fines sent after {band.replace('[', '').replace(')', '').replace(', ', ' to ')} days: {pct(r['collection_rate'])} ended in credit collection ({int(r['cases']):,} fines)"
        for (band, r) in zip(b["band_days"], b.to_dict("records"))], True)
    m, r = col.fit_adjusted(d)
    add("adjusted_effect", "model", "Effect of sending delay after adjustment", [
        f"After adjusting for year, amount, article, vehicle class and points, each extra 30 days before sending multiplies the odds of credit collection by {r['odds_ratio_per_30_days']:.3f} "
        f"(95 percent interval {r['ci_low']:.3f} to {r['ci_high']:.3f}, p = {r['p_value']:.2f})"], True, r)
    s = []
    for name, f in (("ignoring year", col.FORMULA), ("treating year as a number", col.FORMULA + " + start_year")):
        rr = col.fit_adjusted(d, f)[1]
        s.append(f"{name.capitalize()}, the same factor is {rr['odds_ratio_per_30_days']:.3f} (95 percent interval {rr['ci_low']:.3f} to {rr['ci_high']:.3f})")
    add("sensitivity", "model", "Sensitivity to how year is handled", s, True)
    ts = col.time_split_auc(d, 2006)
    add("time_split", "model", "Does the model predict later years", [
        f"A model trained on fines created before 2006 ranks fines from 2006 onward with an AUC of {ts['auc']:.3f} (0.5 is chance); "
        f"the collection rate was {pct(ts['baseline_rate_train'])} before and {pct(ts['baseline_rate_test'])} after"], True, ts)
    sc = col.scenario_faster_sending(d, m, 30)
    add("scenario", "scenario", "What if fines were sent sooner", [
        f"Modelled effect of sending every fine 30 days sooner: {sc['change_pp_median']:+.1f} percentage points on the collection rate "
        f"(90 percent interval {sc['change_pp_p5']:+.1f} to {sc['change_pp_p95']:+.1f}); this assumes the association is causal"], True, sc)
    add("caveat_association", "caveat", "Association only", [
        "These results describe patterns in historical data. They do not show that one step makes another happen, and the data does not say what changed in the process over the years"], True)
    return {"dataset": "Road Traffic Fine Management Process (4TU.ResearchData; de Leoni and Mannhardt)", "findings": F}


if __name__ == "__main__":
    import duckdb
    df = duckdb.connect(sys.argv[1], read_only=True).execute("select * from events").fetchdf()
    json.dump(build(df), open(sys.argv[2], "w"), indent=2, default=str)
    print("wrote", sys.argv[2])
