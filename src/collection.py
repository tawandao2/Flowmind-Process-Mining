"""Which fines end in credit collection, and does a slow 'send fine' step matter?

Rules that keep the analysis honest:
  * Only information known when the fine is SENT is used as a predictor
    (amount, article, vehicle class, points, send delay). Later fields such as
    dismissal, expense or notification type are outcomes or consequences and are left out.
  * Cases too recent to have finished are dropped (censoring cutoff).
  * Results are associations. The scenario assumes the association is causal, which is a strong assumption."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from sklearn.metrics import roc_auc_score

CREATE, SEND, COLLECT, PAY = "Create Fine", "Send Fine", "Send for Credit Collection", "Payment"


def case_features(df):
    g = df.groupby("case_id")
    first = lambda a: df[df["activity"] == a].groupby("case_id")["timestamp"].min()
    out = pd.DataFrame({"created": first(CREATE), "sent": first(SEND), "collected": first(COLLECT), "paid": first(PAY)})
    for col in ("amount", "article", "vehicleClass", "points"):
        if col in df.columns:
            out[col] = g[col].first()
    out["send_delay_days"] = (out["sent"] - out["created"]).dt.total_seconds() / 86400
    out["to_collection_days"] = (out["collected"] - out["created"]).dt.total_seconds() / 86400
    out["start_year"] = out["created"].dt.year
    out["collected_flag"] = out["collected"].notna().astype(int)
    return out.reset_index()


def censoring_cutoff(feat, log_end, q=0.99):
    """Cases created after this date may not have had time to reach collection, so they are excluded."""
    window = feat["to_collection_days"].dropna().quantile(q)
    return log_end - pd.Timedelta(days=float(window)), float(window)


def analysis_set(feat, cutoff, max_delay_q=0.99, min_year_cases=200):
    d = feat[feat["sent"].notna() & (feat["created"] <= cutoff) & (feat["send_delay_days"] >= 0)].copy()
    d = d[d["send_delay_days"] <= d["send_delay_days"].quantile(max_delay_q)]
    # tiny year groups can make a model impossible to fit (one case that was collected = perfect separation)
    d = d[d.groupby("start_year")["case_id"].transform("size") >= min_year_cases].copy()
    d["send_delay_30d"] = d["send_delay_days"] / 30.0
    d["log_amount"] = np.log1p(d["amount"].fillna(d["amount"].median()))
    d["points"] = d["points"].fillna(0)
    for col, k in (("vehicleClass", 3), ("article", 6)):
        top = d[col].astype(str).value_counts().index[:k]
        d[col + "_g"] = d[col].astype(str).where(d[col].astype(str).isin(top), "other")
    return d


FORMULA = "collected_flag ~ send_delay_30d + log_amount + points + C(vehicleClass_g) + C(article_g)"


def delay_bands(d, edges=(0, 30, 60, 90, 120, 10_000)):
    d = d.assign(band=pd.cut(d["send_delay_days"], list(edges), right=False))
    rows = []
    for b, x in d.groupby("band", observed=True):
        lo, hi = proportion_confint(x["collected_flag"].sum(), len(x), method="wilson")
        rows.append({"band_days": str(b), "cases": len(x), "collection_rate": x["collected_flag"].mean(),
                     "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def _fit(formula, d):
    m = smf.logit(formula, d).fit(disp=0, maxiter=200)
    if not m.mle_retvals.get("converged", False):
        raise RuntimeError("logistic model did not converge; its coefficients cannot be trusted")
    return m


def fit_adjusted(d, formula=FORMULA + " + C(start_year)"):
    m = _fit(formula, d)
    ci = m.conf_int().loc["send_delay_30d"]
    return m, {"metric_id": "model:send_delay_odds_ratio", "n": int(m.nobs),
               "odds_ratio_per_30_days": float(np.exp(m.params["send_delay_30d"])),
               "ci_low": float(np.exp(ci[0])), "ci_high": float(np.exp(ci[1])),
               "p_value": float(m.pvalues["send_delay_30d"]),
               "note": "association after adjusting for amount, article, vehicle class, points and year"}


def time_split_auc(d, split_year):
    """Train on earlier years, test on later ones: does the model still rank cases well on unseen years?"""
    tr, te = d[d["start_year"] < split_year], d[d["start_year"] >= split_year]
    m = _fit(FORMULA, tr)
    return {"metric_id": "model:time_split_auc", "train_cases": len(tr), "test_cases": len(te),
            "auc": float(roc_auc_score(te["collected_flag"], m.predict(te))),
            "baseline_rate_train": float(tr["collected_flag"].mean()), "baseline_rate_test": float(te["collected_flag"].mean())}


def scenario_faster_sending(d, m, days_faster=30, n_draws=300, seed=0):
    """Model-based what-if: every fine is sent `days_faster` days sooner (not below 0).
    Uncertainty comes only from the model's coefficients. ASSUMES the association is causal."""
    rng = np.random.default_rng(seed)
    d2 = d.copy()
    d2["send_delay_30d"] = (d["send_delay_days"] - days_faster).clip(lower=0) / 30.0
    draws = rng.multivariate_normal(m.params.values, m.cov_params().values, n_draws)
    from patsy import build_design_matrices
    di = m.model.data.orig_exog.design_info
    Xa = np.asarray(build_design_matrices([di], d)[0])
    Xb = np.asarray(build_design_matrices([di], d2)[0])
    from scipy.special import expit as sig
    diffs = np.array([(sig(Xb @ b) - sig(Xa @ b)).mean() for b in draws])
    return {"metric_id": f"scenario:send_{days_faster}_days_faster", "days_faster": days_faster,
            "current_collection_rate": float(d["collected_flag"].mean()),
            "change_pp_median": float(np.median(diffs) * 100),
            "change_pp_p5": float(np.percentile(diffs, 5) * 100), "change_pp_p95": float(np.percentile(diffs, 95) * 100),
            "assumptions": "association treated as causal; other steps unchanged; uncertainty from model coefficients only"}
