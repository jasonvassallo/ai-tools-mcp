#!/usr/bin/env python3
"""Join blind-judge verdicts back to arms via units/_key.json and aggregate."""

import json
import pathlib
import statistics as st
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).parent


def load_journal(path):
    detect, fp = {}, {}
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("type") != "result":
            continue
        val = rec.get("value") or rec.get("result")
        if not isinstance(val, dict):
            continue
        uid = val.get("unit_id")
        if not uid:
            continue
        if "found_planted_defect" in val:
            detect[uid] = val
        elif "majority_real" in val or "is_real_defect" in val:
            fp.setdefault(uid, []).append(val)
    return detect, fp


def main():
    journal = sys.argv[1]
    key = json.loads((HERE / "units" / "_key.json").read_text())
    detect, fp_raw = load_journal(journal)

    # fp votes may arrive as per-lens records; fold to majority per unit
    fp = {}
    for uid, votes in fp_raw.items():
        if len(votes) == 1 and "majority_real" in votes[0]:
            fp[uid] = votes[0]
            continue
        real = sum(1 for v in votes if v.get("is_real_defect"))
        spec = sum(1 for v in votes if v.get("is_speculation_beyond_diff"))
        sev = sum(1 for v in votes if v.get("severity_justified"))
        fp[uid] = {
            "n_votes": len(votes),
            "majority_real": real * 2 > len(votes),
            "majority_speculative": spec * 2 > len(votes),
            "majority_severity_justified": sev * 2 > len(votes),
        }

    print(f"loaded {len(detect)} detect verdicts, {len(fp)} fp verdicts\n")

    # ── detection on buggy diffs ──
    D = defaultdict(
        lambda: {"hit": 0, "partial": 0, "miss": 0, "unrel": [], "spec": [], "n": 0}
    )
    for uid, v in detect.items():
        k = key.get(uid)
        if not k:
            continue
        a = D[k["arm"]]
        a["n"] += 1
        mq = v.get("match_quality")
        a["hit" if mq == "exact" else ("partial" if mq == "partial" else "miss")] += 1
        a["unrel"].append(v.get("n_findings_unrelated", 0))
        a["spec"].append(v.get("n_findings_speculative", 0))

    print("=== DETECTION on planted-bug diffs (blind-judged) ===")
    print(
        f"{'arm':14} {'n':>3} {'exact':>7} {'partial':>8} {'missed':>7} "
        f"{'unrel/run':>10} {'spec/run':>9}"
    )
    for arm, a in sorted(D.items()):
        n = a["n"] or 1
        print(
            f"{arm:14} {a['n']:3} {a['hit'] / n:7.2f} {a['partial'] / n:8.2f} "
            f"{a['miss'] / n:7.2f} {st.mean(a['unrel']):10.2f} "
            f"{st.mean(a['spec']):9.2f}"
        )

    # ── claimed findings on clean diffs ──
    C = defaultdict(
        lambda: {
            "n": 0,
            "real": 0,
            "spec": 0,
            "sev_ok": 0,
            "by_sev": defaultdict(int),
            "bogus_high": 0,
        }
    )
    for uid, v in fp.items():
        k = key.get(uid)
        if not k:
            continue
        c = C[k["arm"]]
        c["n"] += 1
        sev = k.get("severity", "?")
        c["by_sev"][sev] += 1
        if v.get("majority_real"):
            c["real"] += 1
        if v.get("majority_speculative"):
            c["spec"] += 1
        if v.get("majority_severity_justified"):
            c["sev_ok"] += 1
        if sev in ("CRITICAL", "MAJOR") and not v.get("majority_severity_justified"):
            c["bogus_high"] += 1

    print("\n=== CLAIMED FINDINGS on CLEAN diffs (blind, 3 adversarial judges) ===")
    print(
        f"{'arm':14} {'claims':>7} {'real':>6} {'invented':>9} "
        f"{'specul.':>8} {'sev_ok':>7} {'unjust.HIGH':>12}"
    )
    for arm, c in sorted(C.items()):
        n = c["n"] or 1
        print(
            f"{arm:14} {c['n']:7} {c['real'] / n:6.2f} "
            f"{(c['n'] - c['real']) / n:9.2f} {c['spec'] / n:8.2f} "
            f"{c['sev_ok'] / n:7.2f} {c['bogus_high']:12}"
        )
    for arm, c in sorted(C.items()):
        print(f"   {arm:14} severity mix: {dict(c['by_sev'])}")

    out = {
        "detect": {
            k: dict(
                v,
                unrel=st.mean(v["unrel"]) if v["unrel"] else 0,
                spec=st.mean(v["spec"]) if v["spec"] else 0,
            )
            for k, v in D.items()
        },
        "clean": {
            k: {
                kk: (dict(vv) if isinstance(vv, defaultdict) else vv)
                for kk, vv in v.items()
            }
            for k, v in C.items()
        },
    }
    (HERE / "judge_summary.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
