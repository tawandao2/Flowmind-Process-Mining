import sys, pathlib
import pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))
from load_log import normalise
from process_stats import case_table, variants, transitions, rework

def mk(rows):
    return normalise(pd.DataFrame([{"case:concept:name": c, "concept:name": a, "time:timestamp": t} for c, a, t in rows]))

# A: Received > Checked > Approved (2 days)   B: same path (4 days)   C: has a loop (6 days)
LOG = mk([
    ("A", "Received", "2024-01-01 00:00"), ("A", "Checked", "2024-01-01 12:00"), ("A", "Approved", "2024-01-03 00:00"),
    ("B", "Received", "2024-01-01 00:00"), ("B", "Checked", "2024-01-02 00:00"), ("B", "Approved", "2024-01-05 00:00"),
    ("C", "Received", "2024-01-01 00:00"), ("C", "Checked", "2024-01-02 00:00"), ("C", "Sent back", "2024-01-03 00:00"),
    ("C", "Checked", "2024-01-06 00:00"), ("C", "Approved", "2024-01-07 00:00"),
])

def test_event_ids_are_unique_and_ordered():
    assert LOG.event_id.is_unique and list(LOG[LOG.case_id == "C"].position) == [1, 2, 3, 4, 5]

def test_case_durations():
    d = case_table(LOG).set_index("case_id").duration_days
    assert d["A"] == 2 and d["B"] == 4 and d["C"] == 6

def test_variants_share():
    v = variants(LOG)
    assert v.iloc[0].cases == 2 and abs(v.iloc[0].share - 2 / 3) < 1e-9

def test_transition_wait_is_measured_between_steps():
    t = transitions(LOG).set_index(["activity", "next_activity"])
    assert t.loc[("Sent back", "Checked"), "median_wait_h"] == 72          # 3 days
    assert t.loc[("Checked", "Approved"), "count"] == 3                    # A, B and C's second check

def test_rework_finds_repeated_activity_only():
    r = rework(LOG)
    assert list(r.activity) == ["Checked"] and r.iloc[0].cases == 1 and abs(r.iloc[0].share_of_cases - 1 / 3) < 1e-9

def test_missing_columns_are_rejected():
    import pytest
    with pytest.raises(ValueError):
        normalise(pd.DataFrame({"foo": [1]}))
