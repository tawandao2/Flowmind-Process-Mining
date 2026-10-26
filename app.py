import json, pathlib
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="FlowMind", layout="wide")
S = json.load(open(pathlib.Path(__file__).parent / "data" / "summary.json"))
F = {f["id"]: f for f in S["bundle"]["findings"]}

st.title("FlowMind: where does a process lose time?")
st.caption("Process mining on 150,370 real traffic fines from an Italian local police system, 2000 to 2013. "
           "Data: Road Traffic Fine Management Process, 4TU.ResearchData (de Leoni and Mannhardt).")

st.info("**Headline.** The obvious story is that fines sent faster get collected more often. "
        "That pattern is real in the raw numbers, but it disappears once you account for the year. "
        "So this analysis does not support the claim that sending sooner would help.")

t1, t2, t3, t4 = st.tabs(["Process map", "Where time goes", "The collection question", "Written report"])

with t1:
    tr = pd.DataFrame(S["transitions"])
    minshare = st.slider("Show steps that happen at least this many times", 100, 100000, 5000, step=100)
    keep = tr[tr["count"] >= minshare]
    lines = ["digraph G { rankdir=LR; node [shape=box, style=rounded, fontsize=11]; edge [fontsize=9]; nodesep=0.5; ranksep=1.1;"]
    for r in keep.itertuples():
        w = r.median_wait_h / 24
        lines.append(f'"{r.activity}" -> "{r.next_activity}" [label="{int(r.count):,} fines\\nmedian {w:.0f} days", penwidth={1 + min(r.count / 30000, 4):.1f}];')
    lines.append("}")
    st.graphviz_chart("\n".join(lines))
    st.caption("Thicker arrows mean more fines took that step. Labels show the median wait before the next step.")
    st.subheader("Most common paths")
    st.dataframe(pd.DataFrame(S["variants"]).assign(share=lambda x: (x.share * 100).round(1), median_days=lambda x: x.median_days.round(0)).rename(columns={"variant": "path", "cases": "fines", "median_days": "median days", "share": "share of fines (%)"}),
                 hide_index=True, width="stretch")

with t2:
    tr = pd.DataFrame(S["transitions"])
    tr["share of all waiting"] = tr["total_wait_h"] / tr["total_wait_h"].sum()
    tr["step"] = tr["activity"] + " to " + tr["next_activity"]
    top = tr.head(8)
    st.bar_chart(top.set_index("step")["share of all waiting"])
    st.dataframe(top[["step", "count", "median_wait_h", "p90_wait_h", "share of all waiting"]].assign(
        median_days=lambda x: (x.median_wait_h / 24).round(0), p90_days=lambda x: (x.p90_wait_h / 24).round(0))
        [["step", "count", "median_days", "p90_days", "share of all waiting"]], hide_index=True, width="stretch")
    st.warning(F["fixed_wait"]["facts"][0] + ". Treat that step as a rule to confirm with the owner, not a delay to remove.")
    st.caption("The longest wait, from adding a penalty to credit collection, is partly a legal timeline. "
               "A long wait is not automatically a problem.")

with t3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Raw: slower sending, fewer collections")
        b = pd.DataFrame(S["bands"])
        b["band"] = b["band_days"].str.replace("[", "", regex=False).str.replace(")", "", regex=False).str.replace(", ", " to ").str.replace(" to 10000", "+")
        st.altair_chart(alt.Chart(b).mark_bar().encode(
            x=alt.X("band:N", sort=list(b["band"]), title="Days from fine created to sent (0 to 30 means within 30 days)", axis=alt.Axis(labelAngle=0, labelOverlap=False)),
            y=alt.Y("collection_rate:Q", title="Share ending in credit collection", axis=alt.Axis(format="%"))), width="stretch")
    with c2:
        st.subheader("But the year moved both things")
        y = pd.DataFrame(S["by_year"]); y["year"] = y["year"].astype(str)
        st.altair_chart(alt.Chart(y).mark_line(point=True).encode(
            x=alt.X("year:N", title="Year fine was created", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("rate:Q", title="Share ending in credit collection", axis=alt.Axis(format="%"))), width="stretch")
        st.altair_chart(alt.Chart(y).mark_line(point=True, color="#c0504d").encode(
            x=alt.X("year:N", title="Year fine was created", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("median_delay_days:Q", title="Median days before sending")), width="stretch")
    st.markdown("The median wait before sending rose from about 49 days in 2000 to about 125 days by 2004, while the collection rate fell. So the raw pattern mixes two things together. "
                "Comparing fines within the same year gives a very different answer:")
    a = F["adjusted_effect"]["numbers"]
    st.metric("Odds ratio per extra 30 days before sending (year adjusted)", f"{a['odds_ratio_per_30_days']:.3f}",
              help=f"95 percent interval {a['ci_low']:.3f} to {a['ci_high']:.3f}. 1.0 means no effect.")
    st.write(F["adjusted_effect"]["facts"][0])
    for line in F["sensitivity"]["facts"]:
        st.write("Sensitivity: " + line)
    st.write(F["time_split"]["facts"][0])
    st.write(F["scenario"]["facts"][0])

with t4:
    r = S["report"]
    st.caption("Written by the rules only writer (no AI) so this page needs no key. The Gemini writer produces the same format "
               "and must pass the same checks: every number traced to a cited finding, no causal wording, all caveats included.")
    st.success(f"Checker status: {r['status']}")
    for stmt in r["statements"]:
        st.write(stmt["text"])
        with st.expander("Sources: " + ", ".join(stmt["finding_ids"])):
            for fid in stmt["finding_ids"]:
                st.markdown(f"**{fid}**")
                for fact in F[fid]["facts"]:
                    st.write("• " + fact)
