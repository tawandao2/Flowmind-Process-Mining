"""Plain pandas process statistics. No AI here: these numbers are the evidence."""
import pandas as pd


def case_table(df):
    g = df.groupby("case_id")
    t = g.agg(start=("timestamp", "min"), end=("timestamp", "max"), n_events=("activity", "size"))
    t["duration_days"] = (t["end"] - t["start"]).dt.total_seconds() / 86400
    t["variant"] = g["activity"].agg(" > ".join)
    return t.reset_index()


def variants(df, top=10):
    c = case_table(df)
    v = c.groupby("variant").agg(cases=("case_id", "size"), median_days=("duration_days", "median")).reset_index()
    v["share"] = v["cases"] / len(c)
    return v.sort_values("cases", ascending=False).head(top).reset_index(drop=True)


def transitions(df):
    """Every consecutive pair of steps in a case, with how long the case waited between them."""
    d = df.sort_values(["case_id", "position"]).copy()
    d["next_activity"] = d.groupby("case_id")["activity"].shift(-1)
    d["next_time"] = d.groupby("case_id")["timestamp"].shift(-1)
    d = d.dropna(subset=["next_activity"])
    d["wait_hours"] = (d["next_time"] - d["timestamp"]).dt.total_seconds() / 3600
    out = d.groupby(["activity", "next_activity"]).agg(
        count=("wait_hours", "size"), median_wait_h=("wait_hours", "median"),
        mean_wait_h=("wait_hours", "mean"), p90_wait_h=("wait_hours", lambda s: s.quantile(.9)),
        total_wait_h=("wait_hours", "sum")).reset_index()
    out["metric_id"] = "transition:" + out["activity"] + ">" + out["next_activity"]
    return out.sort_values("total_wait_h", ascending=False).reset_index(drop=True)


def rework(df):
    """Activities that occur more than once in the same case."""
    n = df.groupby(["case_id", "activity"]).size().rename("times").reset_index()
    total = df["case_id"].nunique()
    r = n[n["times"] > 1].groupby("activity").agg(cases=("case_id", "size"), max_repeats=("times", "max")).reset_index()
    r["share_of_cases"] = r["cases"] / total
    r["metric_id"] = "rework:" + r["activity"]
    return r.sort_values("cases", ascending=False).reset_index(drop=True)
