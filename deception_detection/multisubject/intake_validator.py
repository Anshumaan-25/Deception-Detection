"""
Subject-package intake validator — run BEFORE burning GPU hours on a new subject.

A "subject package" is one directory holding the subject's session videos and
(for annotated subjects) their ELAN .eaf files — the shape of
my_videos/00SubjectA_session1/. The validator checks everything that has burned
time in the past, prints a PASS/WARN/FAIL checklist, writes a machine-readable
verdict JSON next to it, and exits non-zero on any FAIL.

Clip-index convention (established by the SubjectA corpus, keep it):
    C001 = the dedicated neutral BASELINE clip → needs NO .eaf
    C002..Cnnn = interview clips → each needs a matching .eaf
    pipeline file_index f ↔ clip C{f+1:03d} (baseline is file_index 0)
Videos/eafs are matched on the C### token anywhere in the filename. Files
without a C### token fall back to sorted order (WARN — rename them).

Checks:
  V1  videos found (≥2: baseline + ≥1 interview), extensions recognized
  V2  baseline clip identifiable (C001 or 'baseline' in the name), unambiguous
  V3  no duplicate clip indices
  E1  every interview video has a matching .eaf (FAIL) / orphan .eafs (WARN)
  E2  every .eaf parses, time slots resolve, every interval has end > start
  E3  label vocabulary ⊆ {Truth, Lie, Neutral} (case-exact; anything else WARN)
  E4  each .eaf carries a non-trivial annotated duration (≥10 s total)
  M1  (optional, needs ffprobe; skip with --no-media-probe) every video is
      readable and has BOTH an audio and a video stream
  M2  (optional) baseline duration ≥60 s (WARN below, FAIL below 30 s — the
      whole calibration doctrine rests on this clip)

Usage:
    python -m multisubject.intake_validator <subject_dir> [--no-media-probe]
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".mts", ".m2ts", ".avi"}
EXPECTED_LABELS = {"Truth", "Lie", "Neutral"}
CLIP_TOKEN = re.compile(r"C(\d{3})", re.IGNORECASE)
BASELINE_MIN_S, BASELINE_WARN_S, MIN_EAF_ANNOTATED_MS = 30.0, 60.0, 10_000


def parse_eaf(path):
    root = ET.parse(path).getroot()
    slots = {t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE"))
             for t in root.iter("TIME_SLOT")}
    return [(a.find("ANNOTATION_VALUE").text,
             slots[a.get("TIME_SLOT_REF1")], slots[a.get("TIME_SLOT_REF2")])
            for a in root.iter("ALIGNABLE_ANNOTATION")]


def _clip_index(name):
    m = CLIP_TOKEN.search(name)
    return int(m.group(1)) if m else None


def _ffprobe(path):
    """(duration_s, has_video, has_audio) or None if ffprobe missing/unreadable."""
    if not shutil.which("ffprobe"):
        return None
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration:stream=codec_type", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return (None, False, False)
        j = json.loads(out.stdout)
        kinds = {s.get("codec_type") for s in j.get("streams", [])}
        dur = float(j.get("format", {}).get("duration", "nan"))
        return (dur, "video" in kinds, "audio" in kinds)
    except Exception:
        return (None, False, False)


class Checklist:
    def __init__(self):
        self.rows = []

    def add(self, code, level, msg):
        self.rows.append({"check": code, "level": level, "message": msg})
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "SKIP": "·"}[level]
        print(f"  {icon} [{level:4s}] {code}: {msg}")

    @property
    def verdict(self):
        if any(r["level"] == "FAIL" for r in self.rows):
            return "FAIL"
        return "WARN" if any(r["level"] == "WARN" for r in self.rows) else "PASS"


def validate_package(subject_dir, media_probe=True):
    d = Path(subject_dir)
    ck = Checklist()
    print(f"[intake] validating subject package: {d}")

    # ── V: videos ────────────────────────────────────────────────────────────
    videos = sorted(p for p in d.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTS)
    if len(videos) >= 2:
        ck.add("V1", "PASS", f"{len(videos)} video file(s) found")
    else:
        ck.add("V1", "FAIL", f"only {len(videos)} video file(s) — need the baseline "
                             f"clip + at least one interview")
    vidx = {}
    unindexed = []
    for v in videos:
        i = _clip_index(v.name)
        if i is not None:
            vidx.setdefault(i, []).append(v)
        else:
            unindexed.append(v)
    if unindexed:
        ck.add("V2", "WARN", f"{len(unindexed)} video(s) without a C### token "
                             f"(will rely on sort order — rename them): "
               + ", ".join(p.name for p in unindexed[:5]))
    baseline = vidx.get(1, []) + [v for v in unindexed if "baseline" in v.name.lower()]
    if len(baseline) == 1:
        ck.add("V2", "PASS", f"baseline clip identified: {baseline[0].name}")
    elif not baseline:
        ck.add("V2", "FAIL", "no baseline clip (C001 or 'baseline' in the name) — "
                             "per-subject calibration is impossible without it")
    else:
        ck.add("V2", "FAIL", f"baseline ambiguous ({len(baseline)} candidates): "
               + ", ".join(p.name for p in baseline))
    dupes = {i: v for i, v in vidx.items() if len(v) > 1}
    if dupes:
        ck.add("V3", "FAIL", "duplicate clip indices: "
               + "; ".join(f"C{i:03d}×{len(v)}" for i, v in sorted(dupes.items())))
    else:
        ck.add("V3", "PASS", "clip indices unique")

    # ── E: ELAN annotations ──────────────────────────────────────────────────
    eafs = sorted(d.rglob("*.eaf"))
    eidx = {}
    for e in eafs:
        i = _clip_index(e.name)
        if i is not None:
            eidx.setdefault(i, []).append(e)
    interviews = sorted(i for i in vidx if i is not None and i != 1)
    missing = [i for i in interviews if i not in eidx]
    if not eafs:
        ck.add("E1", "WARN", "no .eaf files at all — subject can be PROCESSED but "
                             "contributes nothing to validation (no labels, no AUC)")
    elif missing:
        ck.add("E1", "FAIL", "interview clip(s) without a matching .eaf: "
               + ", ".join(f"C{i:03d}" for i in missing))
    else:
        ck.add("E1", "PASS", f"all {len(interviews)} interview clip(s) have an .eaf")
    orphans = [i for i in eidx if i not in vidx and i != 1]
    if orphans:
        ck.add("E1", "WARN", ".eaf(s) without a matching video: "
               + ", ".join(f"C{i:03d}" for i in orphans))
    bad_labels, total_ms = set(), {}
    for e in eafs:
        try:
            iv = parse_eaf(e)
        except Exception as ex:
            ck.add("E2", "FAIL", f"{e.name} does not parse: {ex}")
            continue
        if any(s >= t for _, s, t in iv):
            ck.add("E2", "FAIL", f"{e.name}: interval with end ≤ start")
        bad_labels |= {lab for lab, _, _ in iv if lab not in EXPECTED_LABELS}
        total_ms[e.name] = sum(t - s for _, s, t in iv)
    if eafs and not any(r["check"] == "E2" and r["level"] == "FAIL" for r in ck.rows):
        ck.add("E2", "PASS", f"all {len(eafs)} .eaf file(s) parse with valid intervals")
    if bad_labels:
        ck.add("E3", "WARN", f"labels outside {sorted(EXPECTED_LABELS)}: "
               f"{sorted(bad_labels)} — the scorer only reads Truth/Lie; fix or accept")
    elif eafs:
        ck.add("E3", "PASS", "label vocabulary is exactly Truth/Lie/Neutral")
    thin = {n: ms for n, ms in total_ms.items() if ms < MIN_EAF_ANNOTATED_MS}
    if thin:
        ck.add("E4", "WARN", "thin annotation (<10 s total): "
               + ", ".join(f"{n} ({ms/1000:.1f}s)" for n, ms in thin.items()))
    elif total_ms:
        ck.add("E4", "PASS", "every .eaf carries ≥10 s of annotation")

    # ── M: media probe (optional) ────────────────────────────────────────────
    if not media_probe:
        ck.add("M1", "SKIP", "media probe disabled (--no-media-probe)")
    elif not shutil.which("ffprobe"):
        ck.add("M1", "SKIP", "ffprobe not on PATH — stream/duration checks skipped")
    else:
        for v in videos:
            dur, has_v, has_a = _ffprobe(v)
            if not (has_v and has_a):
                ck.add("M1", "FAIL", f"{v.name}: missing "
                       + ("video" if not has_v else "audio") + " stream / unreadable")
        if not any(r["check"] == "M1" for r in ck.rows):
            ck.add("M1", "PASS", f"all {len(videos)} video(s) readable with A+V streams")
        if len(baseline) == 1:
            dur, _, _ = _ffprobe(baseline[0]) or (None, None, None)
            if dur is None:
                ck.add("M2", "WARN", "baseline duration unknown")
            elif dur < BASELINE_MIN_S:
                ck.add("M2", "FAIL", f"baseline is {dur:.0f}s (<{BASELINE_MIN_S:.0f}s) — "
                                     f"cannot support calibration")
            elif dur < BASELINE_WARN_S:
                ck.add("M2", "WARN", f"baseline is {dur:.0f}s (<{BASELINE_WARN_S:.0f}s "
                                     f"recommended) — learned fits will be thin")
            else:
                ck.add("M2", "PASS", f"baseline duration {dur:.0f}s")

    verdict = ck.verdict
    out = d / "intake_validation.json"
    out.write_text(json.dumps({"subject_dir": str(d), "verdict": verdict,
                               "checks": ck.rows}, indent=2))
    print(f"[intake] verdict: {verdict}  (written to {out.name})")
    return verdict, ck.rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate a subject package before running the cascade.")
    ap.add_argument("subject_dir")
    ap.add_argument("--no-media-probe", action="store_true",
                    help="skip ffprobe stream/duration checks")
    a = ap.parse_args()
    verdict, _ = validate_package(a.subject_dir, media_probe=not a.no_media_probe)
    sys.exit(0 if verdict != "FAIL" else 1)
