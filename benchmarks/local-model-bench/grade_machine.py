#!/usr/bin/env python3
"""Deterministic grading. No LLM judge anywhere in this file.

DELEGATE tasks are graded by executing / parsing the output.
REVIEW tasks get their *mechanical* properties graded here (format compliance,
verdict correctness, severity escalation); semantic matching of findings to the
planted defect is deferred to blind judges in the workflow.
"""

import json
import pathlib
import re
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).parent
TASKS = HERE / "tasks"
OUT = HERE / "out"


# ── helpers ───────────────────────────────────────────────────────────


def strip_fence(text: str) -> str:
    """Pull the first fenced code block, or return the text unfenced."""
    m = re.search(r"```(?:[a-zA-Z]*)\n(.*?)```", text, re.S)
    return m.group(1) if m else text


def first_json(text: str):
    t = strip_fence(text).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    # fall back to the first balanced {...}
    start = t.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except Exception:
                        break
        start = t.find("{", start + 1)
    return None


def run_python(source: str, checks: list[str], timeout: int = 20):
    """Exec model-written code plus assertions in a throwaway subprocess."""
    prog = source + "\n\n" + "\n".join(checks) + "\nprint('ALL_OK')\n"
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "cand.py"
        p.write_text(prog)
        try:
            r = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    # returncode gate (Codex): a solution that prints the sentinel and then
    # crashes — or emits it from its own code — must not pass.
    if r.returncode == 0 and r.stdout.strip().endswith("ALL_OK"):
        return True, ""
    err = (r.stderr or r.stdout).strip().splitlines()
    return False, (err[-1] if err else "no output")[:200]


# ── delegate graders ──────────────────────────────────────────────────


def g_json_schema(task, text):
    obj = first_json(text)
    if obj is None:
        return 0.0, "unparseable JSON"
    if not isinstance(obj, dict):
        return 0.0, "not a JSON object"
    exp = task["expect"]
    # presence-only keys: free-text fields the prompt demands but whose
    # exact wording varies by model (Codex: grade existence, not phrasing).
    presence = task.get("expect_keys", [])
    hits, misses = 0, []
    for k, v in exp.items():
        got = obj.get(k)
        # JSON booleans normalize to their JSON spelling so a model output
        # of `false` matches an expected "false" (review: D03 would have
        # rejected the boolean while accepting the number for API_PORT).
        if isinstance(got, bool):
            got = "true" if got else "false"
        ok = (str(got).strip() == str(v).strip()) if got is not None else False
        if ok:
            hits += 1
        else:
            misses.append(f"{k}={got!r}!={v!r}")
    for k in presence:
        if k in obj and str(obj.get(k)).strip():
            hits += 1
        else:
            misses.append(f"{k}=<missing>")
    total = len(exp) + len(presence)
    return hits / total, "; ".join(misses)


def g_pytest(task, text):
    src = strip_fence(text)
    ok, err = run_python(src, task["tests"])
    return (1.0 if ok else 0.0), err


def g_constraint(task, text):
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    bullets = [ln for ln in lines if ln.strip().startswith("- ")]
    c = task["constraints"]
    problems = []
    score = 0.0
    if len(bullets) == c["bullets"]:
        score += 0.4
    else:
        problems.append(f"{len(bullets)} bullets != {c['bullets']}")
    over = [b for b in bullets if len(b.strip()[2:].split()) > c["max_words"]]
    if not over:
        score += 0.4
    else:
        problems.append(f"{len(over)} bullets over {c['max_words']} words")
    if len(lines) == len(bullets) and bullets:
        score += 0.2  # no preamble / trailer
    else:
        problems.append("extra non-bullet lines")
    return score, "; ".join(problems)


def g_shellcheck(task, text):
    src = strip_fence(text)
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "cand.sh"
        p.write_text(src)
        try:
            r = subprocess.run(
                ["shellcheck", "-S", "warning", str(p)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            sc_ok, sc_note = (r.returncode == 0), r.stdout.strip()[:200]
        except FileNotFoundError:
            sc_ok, sc_note = None, "shellcheck not installed"
        except subprocess.TimeoutExpired:
            sc_ok, sc_note = False, "shellcheck timeout"

        # behavioural check: 3 files + 1 subdir -> must print 3
        work = pathlib.Path(td) / "work"
        work.mkdir()
        for n in ("a", "b", "c"):
            (work / n).write_text("x")
        (work / "sub").mkdir()
        (work / "sub" / "d").write_text("x")
        try:
            r2 = subprocess.run(
                ["bash", str(p), str(work)],
                capture_output=True,
                text=True,
                timeout=20,
                cwd=td,
            )
            # exact final-line match (Codex+Gemini): "13", "30", or an
            # error message containing a 3 must not pass.
            out_lines = [x for x in r2.stdout.strip().splitlines() if x.strip()]
            behaves = (
                r2.returncode == 0 and bool(out_lines) and out_lines[-1].strip() == "3"
            )
            r3 = subprocess.run(
                ["bash", str(p)], capture_output=True, text=True, timeout=20, cwd=td
            )
            rejects = r3.returncode != 0
        except Exception as exc:  # noqa: BLE001
            behaves, rejects = False, False
            sc_note = f"{sc_note} | exec: {exc}"[:200]

    score = 0.0
    if behaves:
        score += 0.5
    if rejects:
        score += 0.25
    if sc_ok is not False:
        score += 0.25
    notes = []
    if not behaves:
        notes.append("wrong file count / exec failed")
    if not rejects:
        notes.append("missing-arg not rejected")
    if sc_ok is False:
        notes.append(f"shellcheck: {sc_note}")
    return score, "; ".join(notes)


def g_redact(task, text):
    leaked = [x for x in task["secrets"] if x in text]
    dropped = [k for k in task["keep"] if k not in text]
    # Deleting secret-bearing lines wholesale used to grade as a perfect
    # redaction (Codex). The instruction is substitution: REDACTED must
    # appear at least once per secret.
    missing_sub = max(0, len(task["secrets"]) - text.count("REDACTED"))
    score = 0.0
    score += 0.5 * (1 - len(leaked) / len(task["secrets"]))
    score += 0.3 * (1 - len(dropped) / len(task["keep"]))
    score += 0.2 * (1 - missing_sub / len(task["secrets"]))
    notes = []
    if leaked:
        notes.append(f"LEAKED {len(leaked)} secret(s)")
    if dropped:
        notes.append(f"dropped non-secret: {dropped}")
    if missing_sub:
        notes.append(f"{missing_sub} secret(s) deleted instead of REDACTED")
    return score, "; ".join(notes)


def g_regex(task, text):
    obj = first_json(text)
    pat = obj.get("pattern") if isinstance(obj, dict) else None
    if not pat:
        m = re.search(r"[\"'](\^.*\$)[\"']", strip_fence(text))
        pat = m.group(1) if m else None
    if not pat:
        return 0.0, "no pattern found"
    try:
        rx = re.compile(pat)
    except re.error as exc:
        return 0.0, f"invalid regex: {exc}"
    # fullmatch only (Gemini): the spec demands whole-string anchoring; the
    # old `or rx.match` fallback silently accepted unanchored patterns.
    good = sum(1 for x in task["match"] if rx.fullmatch(x))
    bad = sum(1 for x in task["nomatch"] if rx.fullmatch(x))
    total = len(task["match"]) + len(task["nomatch"])
    score = (good + (len(task["nomatch"]) - bad)) / total
    notes = []
    if good < len(task["match"]):
        notes.append(f"missed {len(task['match']) - good} must-match")
    if bad:
        notes.append(f"wrongly matched {bad} must-NOT-match")
    return score, "; ".join(notes)


def _body_statement_count(src: str, func_name: str):
    """Statement count of `func_name`'s body, or None if unparseable."""
    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return len(node.body)
    return None


def g_forbid(task, text):
    """Instruction adherence: required tokens present, banned tokens absent."""
    src = strip_fence(text)
    missing = [m for m in task["must"] if m not in src]
    violated = [m for m in task["must_not"] if m in src]
    score = 0.0
    score += 0.4 * (1 - len(missing) / max(len(task["must"]), 1))
    score += 0.6 * (1 - len(violated) / max(len(task["must_not"]), 1))
    notes = []
    if missing:
        notes.append(f"missing {missing}")
    if violated:
        notes.append(f"VIOLATED ban: {violated}")
    # Structural check (Codex): the ban list alone never enforced the
    # "body must be exactly one statement" constraint.
    fn = task.get("one_line_func")
    if fn:
        n = _body_statement_count(src, fn)
        if n is None:
            score *= 0.5
            notes.append("unparseable for one-line check")
        elif n != 1:
            score *= 0.5
            notes.append(f"body has {n} statements, constraint says 1")
    return score, "; ".join(notes)


GRADERS = {
    "json_schema": g_json_schema,
    "pytest": g_pytest,
    "constraint": g_constraint,
    "shellcheck": g_shellcheck,
    "redact": g_redact,
    "regex": g_regex,
    "forbid": g_forbid,
}


# ── review: mechanical properties only ────────────────────────────────

VERDICT_RE = re.compile(r"VERDICT:\s*(APPROVE|REQUEST_CHANGES)", re.I)
FINDING_RE = re.compile(
    r"^\s*[-*]\s*\[?(CRITICAL|MAJOR|MINOR|NONE)\]?\s*:?\s*(.*)$", re.I | re.M
)


def parse_review(text: str):
    vm = VERDICT_RE.search(text)
    verdict = vm.group(1).upper() if vm else None
    findings = []
    for sev, desc in FINDING_RE.findall(text):
        sev = sev.upper()
        if sev == "NONE":
            continue
        desc = desc.strip()
        if desc:
            findings.append({"severity": sev, "text": desc})
    declared_none = bool(re.search(r"^\s*[-*]\s*NONE\s*$", text, re.M))
    # strict format: verdict line present, no prose outside the template
    stray = [
        ln
        for ln in text.strip().splitlines()
        if ln.strip()
        and not ln.strip().startswith(("-", "*"))
        and not VERDICT_RE.match(ln.strip())
        and ln.strip().upper() != "FINDINGS:"
    ]
    return {
        "verdict": verdict,
        "findings": findings,
        "declared_none": declared_none,
        "format_ok": bool(verdict) and not stray,
        "stray_lines": len(stray),
    }


def main():
    tasks = {p.stem: json.loads(p.read_text()) for p in TASKS.glob("*.json")}
    rows = []
    for p in sorted(OUT.glob("*.json")):
        rec = json.loads(p.read_text())
        task = tasks[rec["task_id"]]
        row = {
            k: rec[k] for k in ("task_id", "family", "arm", "model", "think", "trial")
        }
        row["wall_s"] = rec.get("wall_s")
        row["eval_count"] = rec.get("eval_count")
        if rec.get("error"):
            row["error"] = rec["error"]
            rows.append(row)
            continue
        text = rec.get("content", "")
        if task["family"] == "delegate":
            score, note = GRADERS[task["grader"]](task, text)
            row.update(grader=task["grader"], score=round(score, 3), note=note)
        else:
            parsed = parse_review(text)
            row.update(kind=task["kind"], **parsed)
            row["n_findings"] = len(parsed["findings"])
            row["n_high"] = sum(
                1 for f in parsed["findings"] if f["severity"] in ("CRITICAL", "MAJOR")
            )
            expected = "APPROVE" if task["kind"] == "clean" else "REQUEST_CHANGES"
            row["verdict_correct"] = parsed["verdict"] == expected
        rows.append(row)

    (HERE / "machine_scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(f"graded {len(rows)} records -> machine_scores.jsonl")


if __name__ == "__main__":
    main()
