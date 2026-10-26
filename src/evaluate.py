"""Tests the checker itself. Takes the valid report, damages it in known ways, and counts how many damaged
reports the checker catches. A checker that catches nothing would also pass every good report, so this matters."""
import json, sys, re, copy
from investigator import Report, validate, run_investigator, RulesOnlyWriter, nums


def mutations(rep, bundle):
    S = rep["statements"]
    def mk(statements): return Report.model_validate({"statements": statements})
    out = []
    for i, s in enumerate(S):
        m = re.search(r"\d[\d,]*(?:\.\d+)?", s["text"])
        if m:
            raw = m.group(0).rstrip(".")
            bumped = f"{float(raw.replace(',', '')) + 7:.{len(raw.split('.')[1]) if '.' in raw else 0}f}"
            c = copy.deepcopy(S); c[i]["text"] = s["text"].replace(raw, bumped, 1); assert c[i]["text"] != s["text"]
            out.append(("changed a number", mk(c)))
        c = copy.deepcopy(S); c[i]["finding_ids"] = ["made_up_id"]; out.append(("invented finding id", mk(c)))
        c = copy.deepcopy(S); c[i]["text"] += " This shows late sending causes lower collection."; out.append(("causal wording", mk(c)))
        c = copy.deepcopy(S); c[i]["finding_ids"] = [x for x in {f["id"] for f in bundle["findings"]} if x not in s["finding_ids"]][:1]; out.append(("citing the wrong finding", mk(c)))
        c = copy.deepcopy(S); del c[i]
        if any(f["must_cover"] and f["id"] in s["finding_ids"] for f in bundle["findings"]) and not any(set(s["finding_ids"]) & set(x["finding_ids"]) for x in c):
            out.append(("dropping a required finding", mk(c)))
        c = copy.deepcopy(S); c[i]["text"] += " It will definitely cost 4,000,000 euros."; out.append(("invented extra claim", mk(c)))
    return out


def main(bundle_path, out_path):
    bundle = json.load(open(bundle_path))
    base = run_investigator(bundle, RulesOnlyWriter())
    assert base["status"] == "ready", base["problems"]
    res = {}
    for kind, rep in mutations(base, bundle):
        caught = bool(validate(rep, bundle))
        a = res.setdefault(kind, [0, 0]); a[0] += caught; a[1] += 1
    summary = {"clean_report_status": base["status"], "kinds": {k: {"caught": v[0], "tried": v[1]} for k, v in res.items()}}
    summary["total_caught"] = sum(v[0] for v in res.values()); summary["total_tried"] = sum(v[1] for v in res.values())
    json.dump(summary, open(out_path, "w"), indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
