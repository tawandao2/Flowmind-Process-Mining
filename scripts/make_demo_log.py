"""Makes a SYNTHETIC invoice-approval log with a planted rework loop.
For development and tests only. It is not real data and no finding from it should be reported."""
import random, sys
import pandas as pd, pm4py
random.seed(1)
rows, t0 = [], pd.Timestamp("2024-01-01 09:00")
for i in range(300):
    t = t0 + pd.Timedelta(hours=random.randint(0, 24 * 60))
    acts = ["Received", "Checked"]
    if random.random() < 0.3:                       # planted: 30% bounce back for missing info
        acts += ["Sent back", "Checked"]
    acts += ["Approved", "Paid"]
    for a in acts:
        t += pd.Timedelta(hours=random.expovariate(1 / (60 if a == "Checked" and "Sent back" in acts else 8)))
        rows.append({"case:concept:name": f"inv{i}", "concept:name": a, "time:timestamp": t})
df = pd.DataFrame(rows)
pm4py.write_xes(df, sys.argv[1])
print("wrote", sys.argv[1], len(df), "events")
