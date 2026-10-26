"""Builds the one small file the dashboard reads, so the app never needs the raw log."""
import json, sys, duckdb
from process_stats import transitions, variants
import collection as col
from findings import build
from investigator import run_investigator, RulesOnlyWriter

df = duckdb.connect(sys.argv[1], read_only=True).execute("select * from events").fetchdf()
bundle = build(df)
tr = transitions(df)
feat = col.case_features(df); cutoff, _ = col.censoring_cutoff(feat, df["timestamp"].max()); d = col.analysis_set(feat, cutoff)
yr = d.groupby("start_year").agg(mean=("collected_flag", "mean"), size=("collected_flag", "size"), delay=("send_delay_days", "median")).reset_index()
rep = run_investigator(bundle, RulesOnlyWriter())
out = {"bundle": bundle, "report": rep,
       "transitions": tr.drop(columns=["metric_id"]).round(2).to_dict("records"),
       "variants": variants(df, 5).round(3).to_dict("records"),
       "by_year": [{"year": int(r.start_year), "rate": float(r["mean"]), "cases": int(r["size"]), "median_delay_days": float(r["delay"])} for _, r in yr.iterrows()],
       "bands": col.delay_bands(d).round(4).to_dict("records")}
json.dump(out, open(sys.argv[2], "w"), indent=1, default=str)
print("ok", rep["status"], len(out["transitions"]), "transitions,", len(out["by_year"]), "years")
