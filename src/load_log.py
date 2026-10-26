"""Load an XES event log into a clean table.

Every event gets an id (<case_id>#<position>) so later stages can cite the exact
rows behind a finding."""
import gzip, shutil, sys, tempfile, pathlib
import pandas as pd
import pm4py


def read_log(path):
    path = pathlib.Path(path)
    if path.suffix == ".gz":   # pm4py reads plain .xes, so unpack first
        tmp = pathlib.Path(tempfile.mkdtemp()) / path.stem
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = tmp
    log = pm4py.read_xes(str(path))
    return log if isinstance(log, pd.DataFrame) else pm4py.convert_to_dataframe(log)


def normalise(df):
    df = df.rename(columns={"case:concept:name": "case_id", "concept:name": "activity",
                            "time:timestamp": "timestamp", "org:resource": "resource"})
    missing = {"case_id", "activity", "timestamp"} - set(df.columns)
    if missing:
        raise ValueError(f"log is missing required columns: {missing}")
    df["case_id"] = df["case_id"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values(["case_id", "timestamp"], kind="stable").reset_index(drop=True)
    df["position"] = df.groupby("case_id").cumcount() + 1
    df["event_id"] = df["case_id"] + "#" + df["position"].astype(str)
    first = ["event_id", "case_id", "position", "activity", "timestamp"]
    return df[first + [c for c in df.columns if c not in first]]


def load(path):
    return normalise(read_log(path))


if __name__ == "__main__":
    import duckdb
    df = load(sys.argv[1])
    con = duckdb.connect(sys.argv[2])
    con.execute("DROP TABLE IF EXISTS events"); con.execute("CREATE TABLE events AS SELECT * FROM df")
    print(f"{df.case_id.nunique()} cases, {len(df)} events, {df.activity.nunique()} activities")
