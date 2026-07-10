"""
Replication driver — one command that chains the multi-subject workflow around
the (manual, GPU) cascade step.

The full workflow has a GPU step in the middle that this driver cannot run:
    1. INTAKE   validate every subject package               ← this driver
    2. CASCADE  process_recording_session per subject (GPU)  ← manual, at the desktop
    3. SCORE    cross-subject replication scorecard          ← this driver

So run this driver TWICE:
    python -m multisubject.run_replication manifest.json
  • first pass (before the cascade): validates every package with a `package_dir`,
    prints the PASS/WARN/FAIL summary, and reports which subjects still need the
    cascade (their `recording_dir` has no *_recording_calibrated.csv yet).
  • second pass (after the cascade): the same command now finds every
    recording_dir populated and runs the scorecard automatically.

Manifest schema (a superset of the scorecard's — extra keys are ignored there):
    {"subjects": [
      {"name": "SubjectA",
       "package_dir":   "my_videos/00SubjectA_session1",              // for INTAKE
       "recording_dir": "pipeline_system_outputs/REC_SUBJECTA",       // for SCORE
       "elan_dir":      ".../annotated Videos anushree"},             // for SCORE
      ...]}
`package_dir` is optional (skip intake for an already-validated subject);
`recording_dir`+`elan_dir` are needed to be scored.

Exit code: non-zero if any package FAILs intake OR (once scoring runs) the
scorecard cannot be built — so it is safe to gate a shell script on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .intake_validator import validate_package
from .replication_scorecard import main as run_scorecard


def _has_recording(recording_dir):
    d = Path(recording_dir) if recording_dir else None
    return bool(d and d.is_dir() and any(d.glob("*_recording_calibrated.csv")))


def run(manifest_path, out_dir=None, media_probe=True):
    manifest = json.loads(Path(manifest_path).read_text())
    subjects = manifest.get("subjects", [])
    if not subjects:
        print("[driver] manifest has no subjects."); return 1

    # ── Stage 1: INTAKE ───────────────────────────────────────────────────────
    print("═" * 72 + "\n  STAGE 1 — INTAKE VALIDATION\n" + "═" * 72)
    intake, any_fail = {}, False
    for s in subjects:
        pkg = s.get("package_dir")
        if not pkg:
            print(f"[{s['name']}] no package_dir — intake skipped")
            intake[s["name"]] = "SKIPPED"
            continue
        if not Path(pkg).exists():
            print(f"[{s['name']}] package_dir does not exist: {pkg}  → FAIL")
            intake[s["name"]] = "FAIL"; any_fail = True
            continue
        print(f"\n── {s['name']} ─────────────────────────────")
        verdict, _ = validate_package(pkg, media_probe=media_probe)
        intake[s["name"]] = verdict
        any_fail = any_fail or verdict == "FAIL"

    # ── Stage-boundary report ─────────────────────────────────────────────────
    print("\n" + "═" * 72 + "\n  SUBJECT STATUS\n" + "═" * 72)
    ready, pending = [], []
    print(f"  {'subject':16s} {'intake':10s} {'cascade':16s}")
    for s in subjects:
        has = _has_recording(s.get("recording_dir"))
        scored_ok = has and s.get("elan_dir")
        print(f"  {s['name']:16s} {intake.get(s['name'], '—'):10s} "
              f"{'✓ done' if has else '… pending (GPU)':16s}")
        (ready if scored_ok else pending).append(s["name"])

    if any_fail:
        print("\n[driver] ✗ at least one package FAILED intake — fix before the "
              "cascade. Not scoring.")
        return 1
    if pending:
        print(f"\n[driver] {len(pending)} subject(s) still need the cascade (GPU): "
              + ", ".join(pending))
        print("[driver] run process_recording_session for those, then re-run this "
              "command to score.")
    if len(ready) < 2:
        print(f"\n[driver] only {len(ready)} subject(s) ready to score (<2) — "
              "scorecard needs ≥2. Stopping after intake.")
        return 0

    # ── Stage 3: SCORE ────────────────────────────────────────────────────────
    print("\n" + "═" * 72 + f"\n  STAGE 3 — REPLICATION SCORECARD ({len(ready)} subjects)\n"
          + "═" * 72)
    scored = {"subjects": [s for s in subjects if s["name"] in ready]}
    tmp = Path(out_dir or Path(manifest_path).parent) / "_scored_manifest.json"
    tmp.write_text(json.dumps(scored))
    csv_path = run_scorecard(str(tmp), out_dir)
    return 0 if csv_path else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Drive the multi-subject replication workflow (intake → [manual cascade] → scorecard).")
    ap.add_argument("manifest")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-media-probe", action="store_true")
    a = ap.parse_args()
    sys.exit(run(a.manifest, a.out, media_probe=not a.no_media_probe))
