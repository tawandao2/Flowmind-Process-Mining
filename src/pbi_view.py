"""A Power BI style report page for FlowMind: slicers, KPI cards and linked visuals."""
import altair as alt
import pandas as pd
import streamlit as st

BLUE, NAVY, ORANGE, YELLOW, RED = "#118DFF", "#12239E", "#E66C37", "#D9B300", "#D64550"

CSS = """
<style>
.stApp {background:#F3F2F1;}
.block-container {padding-top:1.2rem; max-width:1400px;}
.pbi-title {background:#12239E; color:white; padding:14px 22px; border-radius:4px; display:flex; justify-content:space-between; align-items:center;}
.pbi-title h2 {margin:0; padding:0; font-size:1.35rem !important; color:white !important; font-weight:600;}
.pbi-title span {font-size:.8rem; opacity:.85;}
.card {min-height:112px; background:white; border:1px solid #E1DFDD; border-radius:4px; padding:12px 16px; box-shadow:0 1px 2px rgba(0,0,0,.06);}
.card .lab {font-size:.78rem; color:#605E5C; margin-bottom:2px;}
.card .val {font-size:2rem; font-weight:600; color:#252423; line-height:1.15;}
.card .sub {font-size:.75rem; color:#605E5C;}
[class*="st-key-vis_"] {background:#FFFFFF !important; border:1px solid #E1DFDD !important; border-radius:4px !important; box-shadow:0 1px 2px rgba(0,0,0,.06);}
.vt {font-size:.92rem; font-weight:600; color:#252423; margin-bottom:2px;}
.note {font-size:.78rem; color:#605E5C;}
</style>
"""
_n = [0]


def card(col, label, value, sub=""):
    col.markdown(f'<div class="card"><div class="lab">{label}</div><div class="val">{value}</div><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def vis(title):
    _n[0] += 1
    box = st.container(border=True, key=f"vis_{_n[0]}")
    box.markdown(f'<div class="vt">{title}</div>', unsafe_allow_html=True)
    return box


def style(ch, h=240):
    return ch.properties(height=h, background="#FFFFFF").configure_view(strokeWidth=0).configure_axis(
        labelColor="#605E5C", titleColor="#605E5C", gridColor="#EDEBE9", domainColor="#C8C6C4").configure_legend(orient="top", title=None)


def render(S, F, checker):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown('<div class="pbi-title"><h2>Traffic Fines Process Report</h2><span>150,370 real fines, Italian local police, 2000 to 2013 &nbsp;|&nbsp; 4TU.ResearchData</span></div>', unsafe_allow_html=True)
    st.write("")
    yr = pd.DataFrame(S["by_year"])
    s1, s2 = st.columns(2)
    y0, y1 = s1.select_slider("Year the fine was created", options=list(yr.year), value=(int(yr.year.min()), int(yr.year.max())))
    minn = s2.select_slider("Show steps that happen at least this many times", options=[100, 1000, 5000, 20000, 50000], value=1000)
    sel = yr[(yr.year >= y0) & (yr.year <= y1)]
    n = int(sel.cases.sum())
    rate = float((sel.rate * sel.cases).sum() / n)
    delay = float((sel.median_delay_days * sel.cases).sum() / n)
    a = F["adjusted_effect"]["numbers"]

    k = st.columns(5)
    card(k[0], "Fines in selected years", f"{n:,}", f"of {int(yr.cases.sum()):,} with a send date")
    card(k[1], "Ended in credit collection", f"{rate * 100:.1f}%", f"all years {float((yr.rate * yr.cases).sum() / yr.cases.sum()) * 100:.1f}%")
    card(k[2], "Typical days before sending", f"{delay:.0f}", "average of yearly medians")
    card(k[3], "Effect of sending later, year adjusted", f"{a['odds_ratio_per_30_days']:.3f}", f"interval {a['ci_low']:.3f} to {a['ci_high']:.3f}, 1.0 is no effect")
    card(k[4], "Checker catches on broken reports", f"{checker['total_caught']} of {checker['total_tried']}", "deliberate errors")
    st.write("")

    c1, c2, c3 = st.columns([5, 4, 4])
    with c1:
        box = vis("Collection rate and sending delay by year")
        base = alt.Chart(sel).encode(x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0)))
        bars = base.mark_bar(color="#F3C9B7", cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            y=alt.Y("median_delay_days:Q", title="Median days before sending", axis=alt.Axis(titleColor=ORANGE)), tooltip=["year", alt.Tooltip("median_delay_days:Q", format=".0f")])
        line = base.mark_line(color=BLUE, point=alt.OverlayMarkDef(color=BLUE, size=60), strokeWidth=3).encode(
            y=alt.Y("rate:Q", title="Share collected", axis=alt.Axis(format="%", titleColor=BLUE), scale=alt.Scale(domain=[0, 1])),
            tooltip=["year", alt.Tooltip("rate:Q", format=".1%"), "cases"])
        box.altair_chart(style(alt.layer(bars, line).resolve_scale(y="independent")), width="stretch")
    with c2:
        box = vis("Raw view: later sending, fewer collections")
        b = pd.DataFrame(S["bands"])
        b["band"] = b["band_days"].str.replace("[", "", regex=False).str.replace(")", "", regex=False).str.replace(", ", " to ").str.replace(" to 10000", "+")
        ch = alt.Chart(b).mark_bar(color=NAVY, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
            x=alt.X("band:N", sort=list(b["band"]), title="Days before sending", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("collection_rate:Q", title=None, axis=alt.Axis(format="%")), tooltip=["band", alt.Tooltip("collection_rate:Q", format=".1%"), "cases"])
        lab = ch.mark_text(dy=-6, color="#252423", fontSize=11).encode(text=alt.Text("collection_rate:Q", format=".0%"))
        box.altair_chart(style(ch + lab), width="stretch")
    with c3:
        box = vis("The same effect once the year is accounted for")
        o = pd.DataFrame([{"model": "Ignoring year", "or": 0.833, "lo": 0.824, "hi": 0.842},
                          {"model": "Year as a number", "or": 0.901, "lo": 0.890, "hi": 0.912},
                          {"model": "Year adjusted (final)", "or": a["odds_ratio_per_30_days"], "lo": a["ci_low"], "hi": a["ci_high"]}])
        order = list(o.model)
        err = alt.Chart(o).mark_rule(strokeWidth=3, color=NAVY).encode(y=alt.Y("model:N", sort=order, title=None), x=alt.X("lo:Q", title="Odds ratio per extra 30 days", scale=alt.Scale(domain=[0.8, 1.05])), x2="hi:Q")
        pts = alt.Chart(o).mark_point(filled=True, size=110, color=ORANGE).encode(y=alt.Y("model:N", sort=order), x="or:Q", tooltip=["model", alt.Tooltip("or:Q", format=".3f")])
        one = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(color=RED, strokeDash=[4, 3]).encode(x="x:Q")
        box.altair_chart(style(err + pts + one), width="stretch")
        box.markdown('<div class="note">Red line = no effect. Only the year adjusted interval reaches it.</div>', unsafe_allow_html=True)

    st.write("")
    d1, d2, d3 = st.columns([5, 4, 4])
    with d1:
        box = vis("Where the waiting happens (share of all elapsed time)")
        tr = pd.DataFrame(S["transitions"])
        tr = tr[tr["count"] >= minn].copy()
        tr["share"] = tr["total_wait_h"] / pd.DataFrame(S["transitions"])["total_wait_h"].sum()
        tr["step"] = tr["activity"] + " to " + tr["next_activity"]
        tr = tr.sort_values("share", ascending=False).head(7)
        ch = alt.Chart(tr).mark_bar(color=BLUE).encode(y=alt.Y("step:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
                                                      x=alt.X("share:Q", title=None, axis=alt.Axis(format="%")),
                                                      tooltip=["step", alt.Tooltip("share:Q", format=".1%"), "count", alt.Tooltip("median_wait_h:Q", format=",.0f")])
        box.altair_chart(style(ch, 260), width="stretch")
    with d2:
        box = vis("Most common paths (share of fines)")
        v = pd.DataFrame(S["variants"])
        names = {"Create Fine > Send Fine > Insert Fine Notification > Add penalty > Send for Credit Collection": "Full path to collection",
                 "Create Fine > Payment": "Paid straight away", "Create Fine > Send Fine": "Sent, no outcome yet",
                 "Create Fine > Send Fine > Insert Fine Notification > Add penalty > Payment": "Paid after penalty",
                 "Create Fine > Send Fine > Insert Fine Notification > Add penalty > Payment > Payment": "Paid in two parts"}
        v["path"] = v.variant.map(names).fillna(v.variant)
        ch = alt.Chart(v).mark_bar(color=YELLOW).encode(y=alt.Y("path:N", sort="-x", title=None, axis=alt.Axis(labelLimit=240)),
                                                        x=alt.X("share:Q", title=None, axis=alt.Axis(format="%")),
                                                        tooltip=["variant", alt.Tooltip("share:Q", format=".1%"), alt.Tooltip("median_days:Q", format=".0f"), "cases"])
        box.altair_chart(style(ch, 260), width="stretch")
    with d3:
        box = vis("Why a model cannot help: the rate shifted over time")
        t = F["time_split"]["numbers"]
        ts = pd.DataFrame([{"period": "Before 2006", "rate": t["baseline_rate_train"]}, {"period": "2006 onward", "rate": t["baseline_rate_test"]}])
        ch = alt.Chart(ts).mark_bar(color=NAVY, cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(x=alt.X("period:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                                                                                                    y=alt.Y("rate:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])))
        lab = ch.mark_text(dy=-6, color="#252423", fontSize=12).encode(text=alt.Text("rate:Q", format=".1%"))
        box.altair_chart(style(ch + lab, 215), width="stretch")
        box.markdown(f'<div class="note">A model trained on the earlier fines ranks later ones with AUC {t["auc"]:.3f}, where 0.5 is chance.</div>', unsafe_allow_html=True)
    st.caption("The year slicer filters the year chart and the first three cards. The step slicer filters the waiting chart. The odds ratio, path and time split visuals use the full data.")
