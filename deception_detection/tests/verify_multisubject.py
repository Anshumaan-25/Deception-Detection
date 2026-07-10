"""
verify_multisubject.py — intake validator + replication scorecard checks
(pure pandas/numpy/stdlib, synthetic fixtures; no GPU, no real footage).

Covers:
  1. intake validator: a well-formed package PASSes; each failure class fires
     the right check (missing baseline, missing eaf, corrupt eaf, duplicate
     clip index, too few videos); label/annotation problems WARN not FAIL;
     an unannotated subject is processable (WARN, not FAIL); verdict JSON written
  2. replication scorecard against a planted 3-subject world:
     a channel that separates in all subjects → REPLICATES;
     a one-subject channel → SUBJECT-SPECIFIC (the noise-sign pigeonhole trap
     must NOT rescue it via direction);
     consistent-direction/weak-magnitude → DIRECTION-ONLY;
     pure noise → NO-SIGNAL; all-NaN → INSUFFICIENT-DATA;
     baseline clips and sub-0.6-overlap windows never scored
Run: python tests/verify_multisubject.py
"""
import json, os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multisubject.intake_validator import validate_package
from multisubject.replication_scorecard import build_scorecard, main as scorecard_main
from multisubject.run_replication import run as run_driver

ok = 0
def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("  ✓", msg)

EAF = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
 <TIME_ORDER>
  <TIME_SLOT TIME_SLOT_ID="t1" TIME_VALUE="0"/><TIME_SLOT TIME_SLOT_ID="t2" TIME_VALUE="48000"/>
  <TIME_SLOT TIME_SLOT_ID="t3" TIME_VALUE="50000"/><TIME_SLOT TIME_SLOT_ID="t4" TIME_VALUE="98000"/>
 </TIME_ORDER>
 <TIER TIER_ID="gt">
  <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="t1" TIME_SLOT_REF2="t2">
    <ANNOTATION_VALUE>Truth</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
  <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="t3" TIME_SLOT_REF2="t4">
    <ANNOTATION_VALUE>Lie</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
 </TIER>
</ANNOTATION_DOCUMENT>"""


def make_package(root, name="SubjectX", videos=("C001", "C002", "C003"),
                 eafs=("C002", "C003"), eaf_body=EAF):
    d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
    for tok in videos:
        open(os.path.join(d, f"B09{tok}_interview.mp4"), "wb").write(b"\x00")
    for tok in eafs:
        open(os.path.join(d, f"B09{tok}_annot.eaf"), "w").write(eaf_body)
    return d


def levels(rows):
    return {r["check"]: r["level"] for r in rows}


print("1. intake validator")
with tempfile.TemporaryDirectory() as tmp:
    d = make_package(tmp)
    v, rows = validate_package(d, media_probe=False)
    check(v == "PASS", "well-formed package → PASS")
    check(os.path.exists(os.path.join(d, "intake_validation.json")),
          "machine-readable verdict JSON written next to the package")

    v, rows = validate_package(make_package(tmp, "NoBase", videos=("C002", "C003")),
                               media_probe=False)
    check(v == "FAIL" and levels(rows)["V2"] == "FAIL", "missing baseline (no C001) → V2 FAIL")

    v, rows = validate_package(make_package(tmp, "NoEaf", eafs=("C002",)), media_probe=False)
    check(v == "FAIL" and levels(rows)["E1"] == "FAIL",
          "interview without a matching .eaf → E1 FAIL")

    v, rows = validate_package(make_package(tmp, "Corrupt", eaf_body="<not xml"),
                               media_probe=False)
    check(v == "FAIL", "corrupt .eaf → FAIL (E2)")

    v, rows = validate_package(
        make_package(tmp, "BadLabel", eaf_body=EAF.replace(">Lie<", ">Deception<")),
        media_probe=False)
    check(v == "WARN" and levels(rows)["E3"] == "WARN",
          "unknown label vocabulary → WARN (fixable), not FAIL")

    v, rows = validate_package(make_package(tmp, "NoAnnot", eafs=()), media_probe=False)
    check(v == "WARN" and levels(rows)["E1"] == "WARN",
          "unannotated subject → processable (WARN), contributes no validation")

    dup = make_package(tmp, "Dup")
    open(os.path.join(dup, "B09C002_retake.mp4"), "wb").write(b"\x00")
    v, rows = validate_package(dup, media_probe=False)
    check(v == "FAIL" and levels(rows)["V3"] == "FAIL", "duplicate clip index → V3 FAIL")

    v, rows = validate_package(make_package(tmp, "OneVid", videos=("C001",), eafs=()),
                               media_probe=False)
    check(v == "FAIL" and levels(rows)["V1"] == "FAIL", "single video → V1 FAIL")


# ── planted 3-subject world for the scorecard ────────────────────────────────
CHANNELS = ["AU12_velocity_max",   # replicates: strong in all 3, direction +
            "gaze_x_mean",         # subject-1 quirk: strong only in S1
            "blink_rate",          # consistent direction, weak magnitude
            "wavlm_latent_0",      # pure noise
            "ear_mean"]            # all-NaN → insufficient

def make_subject(root, name, quirk=False, rng=None, baseline_src_idx=None):
    d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
    elan = os.path.join(d, "elan"); os.makedirs(elan, exist_ok=True)
    if baseline_src_idx is not None:
        json.dump({"feature_means": {}, "feature_stds": {}, "baseline_window_count": 1,
                   "source_csv": f"/store/REC_{name}_{baseline_src_idx:03d}_windowed_features.csv"},
                  open(os.path.join(d, f"REC_{name}_baseline_stats.json"), "w"))
    rows = []
    for fidx in range(3):                       # 0 = baseline, 1..2 interviews
        for w in range(100):
            t = w * 1000.0
            is_lie = 50000 <= t < 98000         # matches the EAF intervals
            r = {"window_id": w, "start_time_ms": t, "end_time_ms": t + 2000,
                 "file_index": fidx, "clip_window_id": w}
            r["AU12_velocity_max"] = rng.normal(2.0 if is_lie and fidx else 0.3, 0.5)
            r["gaze_x_mean"] = rng.normal(-2.0 if quirk and is_lie and fidx else 0.0, 0.5)
            r["blink_rate"] = rng.normal(0.4 if is_lie and fidx else 0.0, 1.0)
            r["wavlm_latent_0"] = rng.normal(0.0, 1.0)
            r["ear_mean"] = np.nan
            rows.append(r)
        if fidx:                                # interviews get the eaf
            open(os.path.join(elan, f"B09C{fidx+1:03d}_gt.eaf"), "w").write(EAF)
    pd.DataFrame(rows).to_csv(os.path.join(d, f"REC_{name}_recording_calibrated.csv"),
                              index=False)
    return {"name": name, "recording_dir": d, "elan_dir": elan}


print("2. replication scorecard")
with tempfile.TemporaryDirectory() as tmp:
    # seed 5: verified to satisfy the planted-world properties below (with 94
    # windows/class, a noise AUC crosses 0.60 by chance for ~1 seed in 6 — at
    # seed 0 it did, and the scorecard HONESTLY called it SUBJECT-SPECIFIC;
    # the fixture, not the logic, has to guarantee "no signal anywhere")
    rng = np.random.default_rng(5)
    manifest = {"subjects": [make_subject(tmp, "S1", quirk=True, rng=rng),
                             make_subject(tmp, "S2", rng=rng),
                             make_subject(tmp, "S3", rng=rng)]}
    mpath = os.path.join(tmp, "manifest.json")
    json.dump(manifest, open(mpath, "w"))
    card, subjects = build_scorecard(manifest)
    v = dict(zip(card.channel, card.verdict))
    check(all(subjects[s]["wavlm_latent_0"]["auc"] < 0.60 for s in subjects),
          "fixture property holds: the noise channel shows no signal in ANY subject")
    check(v["AU12_velocity_max"] == "REPLICATES",
          "channel strong in all subjects → REPLICATES")
    check(v["gaze_x_mean"] == "SUBJECT-SPECIFIC",
          "one-subject channel → SUBJECT-SPECIFIC (noise signs don't rescue it)")
    check(v["blink_rate"] == "DIRECTION-ONLY",
          "consistent direction, weak magnitude → DIRECTION-ONLY")
    check(v["wavlm_latent_0"] == "NO-SIGNAL", "pure noise → NO-SIGNAL")
    check(v["ear_mean"] == "INSUFFICIENT-DATA", "all-NaN channel → INSUFFICIENT-DATA")
    check(bool(card[card.channel == "AU12_velocity_max"].tracked.iloc[0]),
          "SubjectA tracked channel flagged in the scorecard")
    s1 = subjects["S1"]
    check(s1["AU12_velocity_max"]["n_lie"] >= 30 and s1["AU12_velocity_max"]["n_truth"] >= 30,
          "pure-window counts clear the pre-registered adequacy floor")
    csv_path = scorecard_main(mpath)
    check(csv_path and os.path.exists(csv_path), "scorecard CSV written")
    out = pd.read_csv(csv_path)
    check({"auc_S1", "auc_S2", "auc_S3"} <= set(out.columns),
          "CSV carries one AUC column per subject")

print("3. scorecard recovers a NON-ZERO baseline index (never assumes 0)")
with tempfile.TemporaryDirectory() as tmp:
    from multisubject.replication_scorecard import load_labeled_windows
    rng = np.random.default_rng(1)
    # baseline declared as clip index 1 (which DOES carry an eaf) → its labeled
    # windows must be excluded despite having labels
    s = make_subject(tmp, "SB", rng=rng, baseline_src_idx=1)
    pure = load_labeled_windows(s["recording_dir"], s["elan_dir"])
    check(not pure.empty and (pure.file_index == 1).sum() == 0,
          "windows from the declared baseline clip (index 1) are not scored")
    check((pure.file_index == 2).sum() > 0, "the true interview clip (index 2) is scored")

print("4. run_replication driver (intake → [manual cascade] → scorecard)")
with tempfile.TemporaryDirectory() as tmp:
    rng = np.random.default_rng(2)
    subs = [make_subject(tmp, "D1", quirk=True, rng=rng),
            make_subject(tmp, "D2", rng=rng)]
    # add package_dir (a well-formed intake package) to each
    for sub in subs:
        pkg = make_package(tmp, sub["name"] + "_pkg")
        sub["package_dir"] = pkg
    manifest = {"subjects": subs}
    mpath = os.path.join(tmp, "drive.json"); json.dump(manifest, open(mpath, "w"))
    rc = run_driver(mpath, media_probe=False)
    check(rc == 0, "driver returns 0 when packages pass intake and recordings are ready")
    check(os.path.exists(os.path.join(tmp, "replication_scorecard.csv")),
          "driver produced the scorecard in the second (post-cascade) state")
    # a FAILing package (missing baseline) blocks scoring, non-zero exit
    bad = make_subject(tmp, "D3", rng=rng)
    bad["package_dir"] = make_package(tmp, "D3_pkg", videos=("C002", "C003"))  # no baseline
    manifest["subjects"].append(bad); json.dump(manifest, open(mpath, "w"))
    check(run_driver(mpath, media_probe=False) == 1,
          "driver returns non-zero and withholds scoring when any package FAILs intake")

print(f"\nverify_multisubject: {ok} checks passed — no GPU, no real footage.")
