#!/usr/bin/env python3
"""Build BLINDED judging units for the review half.

Model identity is stripped from every unit. Units are keyed by an opaque id;
the mapping back to arms lives in units/_key.json, which judges never read.

Two unit types:
  detect  - BUGGY diff: did the candidate's findings identify the planted defect?
  fp      - CLEAN diff: is this specific claimed finding a real defect, or invented?

The `fp` unit is deliberately two-sided: a judge that finds a genuine defect in
a diff I labelled clean must say so. That keeps a flawed corpus from being
scored as model hallucination.
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "tasks"
OUT = HERE / "out"
UNITS = HERE / "units"


def main():
    UNITS.mkdir(parents=True, exist_ok=True)
    for stale in UNITS.glob("*.json"):
        stale.unlink()

    tasks = {p.stem: json.loads(p.read_text()) for p in TASKS.glob("*.json")}
    scores = [
        json.loads(ln)
        for ln in (HERE / "machine_scores.jsonl").read_text().splitlines()
        if ln
    ]

    key = {}
    n = 0
    for row in scores:
        if row["family"] != "review" or row.get("error"):
            continue
        task = tasks[row["task_id"]]
        diff = task["diff"].strip()

        if row["kind"] == "buggy":
            n += 1
            uid = f"U{n:04d}"
            (UNITS / f"{uid}.json").write_text(
                json.dumps(
                    {
                        "unit_id": uid,
                        "type": "detect",
                        "diff": diff,
                        "planted_defect": task["ground_truth"],
                        "candidate_verdict": row["verdict"],
                        "candidate_findings": row["findings"],
                    },
                    indent=2,
                )
            )
            key[uid] = {
                "arm": row["arm"],
                "task_id": row["task_id"],
                "trial": row["trial"],
                "type": "detect",
            }
        else:
            # one unit per claimed finding on a clean diff
            for idx, f in enumerate(row["findings"]):
                n += 1
                uid = f"U{n:04d}"
                (UNITS / f"{uid}.json").write_text(
                    json.dumps(
                        {
                            "unit_id": uid,
                            "type": "fp",
                            "diff": diff,
                            "why_labelled_clean": task["ground_truth"],
                            "claimed_severity": f["severity"],
                            "claimed_finding": f["text"],
                        },
                        indent=2,
                    )
                )
                key[uid] = {
                    "arm": row["arm"],
                    "task_id": row["task_id"],
                    "trial": row["trial"],
                    "type": "fp",
                    "severity": f["severity"],
                    "finding_idx": idx,
                }

    (UNITS / "_key.json").write_text(json.dumps(key, indent=2))
    d = sum(1 for v in key.values() if v["type"] == "detect")
    f = sum(1 for v in key.values() if v["type"] == "fp")
    print(f"wrote {n} blinded units ({d} detect, {f} fp) -> {UNITS}")


if __name__ == "__main__":
    main()
