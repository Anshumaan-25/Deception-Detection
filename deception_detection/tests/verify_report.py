"""
verify_report.py — analyst report generator checks (pure pandas/numpy + stdlib,
synthetic recording fixture; no GPU, no real footage, no network).

Covers the contract of report/analyst_report.py + render_html.py:
  1. assembly integrity (clips, node table, flags == p95 rule)
  2. dead / uncalibratable channels surfaced in quality, never plotted as zeros
  3. baseline-health gate (healthy vs the degenerate near-zero signature)
  4. coupling lane conditionality (healthy fit → shown; degenerate → withheld;
     absent file → 'not run')
  5. ELAN strictly validation-mode (absent by default; overlaid only with --elan-dir)
  6. rendered HTML is self-contained (no external URLs) and JSON-clean (no NaN)
Run: python tests/verify_report.py
"""
import json, os, sys, tempfile
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report.analyst_report import build_report_data, generate_report, FLAG_PERCENTILE

ok = 0
def check(cond, msg):
    global ok
    assert cond, "FAIL: " + msg
    ok += 1
    print("  ✓", msg)

FEATS = ["AU12_velocity_max", "AU12_velocity_tremor_band_power",
         "left_hand_face_distance_min", "right_wrist_velocity_max",
         "gaze_x_mean", "gaze_entropy", "blink_rate", "ear_mean",
         "acoustic_energy_rms", "prosodic_velocity", "mismatch_ratio",
         "head_pitch_tremor_band_power", "macro_motion_energy_mean",
         "AU6_mean", "wavlm_latent_0", "silent_speech_duration_ms"]

EAF = """<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
 <TIME_ORDER>
  <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="5000"/><TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="15000"/>
  <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="30000"/><TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="45000"/>
 </TIME_ORDER>
 <TIER TIER_ID="truth">
  <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a1" TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
    <ANNOTATION_VALUE>Truth</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
  <ANNOTATION><ALIGNABLE_ANNOTATION ANNOTATION_ID="a2" TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
    <ANNOTATION_VALUE>Lie</ANNOTATION_VALUE></ALIGNABLE_ANNOTATION></ANNOTATION>
 </TIER>
</ANNOTATION_DOCUMENT>"""


def make_fixture(root, rid="REC_TEST", degenerate_baseline=False,
                 coupling="healthy"):
    rec = os.path.join(root, rid); os.makedirs(rec, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for fidx in range(3):
        for w in range(60):
            r = {"window_id": w, "start_time_ms": w*1000.0, "end_time_ms": w*1000.0+2000,
                 "frame_count": 60, "cumulative_confidence": 0.85,
                 "file_index": fidx, "clip_window_id": w}
            for c in FEATS:
                if c in ("ear_mean", "silent_speech_duration_ms"):
                    r[c] = np.nan                                   # dead channels
                else:
                    scale = 0.05 if degenerate_baseline else 1.0
                    r[c] = rng.normal() * scale
            # clip 2: an injected freeze+leakage episode (windows 20..29)
            if fidx == 2 and 20 <= w < 30 and not degenerate_baseline:
                r["gaze_x_mean"] = -3.0 + 0.2*rng.normal()          # freeze (suppressed)
                r["AU12_velocity_max"] = 3.5 + 0.2*rng.normal()     # leakage (elevated)
            rows.append(r)
    df = pd.DataFrame(rows)
    z = df[FEATS].to_numpy(dtype=float)
    df["deviation_magnitude"] = np.sqrt(np.nansum(z**2, axis=1))
    df["deviation_percentile"] = df["deviation_magnitude"].rank(pct=True, na_option="keep")
    df.to_csv(os.path.join(rec, f"{rid}_recording_calibrated.csv"), index=False)

    stats = {"feature_means": {c: 0.0 for c in FEATS},
             "feature_stds": {c: (None if c == "silent_speech_duration_ms" else 1.0)
                              for c in FEATS},
             "baseline_window_count": 60, "source_csv": f"{rid}_000_windowed.csv"}
    with open(os.path.join(rec, f"{rid}_baseline_stats.json"), "w") as f:
        json.dump(stats, f)

    if coupling in ("healthy", "degenerate"):
        os.makedirs(os.path.join(rec, "coupling_fit"), exist_ok=True)
        with open(os.path.join(rec, "coupling_fit", "fit_report.json"), "w") as f:
            json.dump({"coupling_ratio": 0.35 if coupling == "healthy" else 0.95}, f)
        crows = []
        for fidx in range(3):
            for w in range(60):
                crows.append({"file_index": fidx, "start_time_ms": w*1000.0,
                              "end_time_ms": w*1000.0+2000,
                              "coupling_z_gaze": (4.0 if fidx == 2 and 20 <= w < 30 else 0.1),
                              "coupling_z_au_mouth": 0.2, "coupling_z_voice": np.nan})
        pd.DataFrame(crows).to_csv(os.path.join(rec, "coupling_attribution.csv"), index=False)
    return rec, df


print("1. assembly integrity")
with tempfile.TemporaryDirectory() as tmp:
    rec, df = make_fixture(tmp)
    data = build_report_data(rec, generated_on="2026-07-10")
    check(data["meta"]["recording_id"] == "REC_TEST", "recording id inferred from filename")
    check(data["meta"]["n_clips"] == 3 and data["clips"][0]["is_baseline"],
          "3 clips, file_index 0 marked baseline")
    expected_flags = int((df["deviation_percentile"] >= FLAG_PERCENTILE).sum())
    got_flags = sum(len(c["flags"]) for c in data["clips"])
    check(got_flags == expected_flags,
          f"flag count matches the p95 rule ({got_flags} == {expected_flags})")
    check(len(data["node_table"]) == 3 and len(data["node_table"][0]["cells"]) == 11,
          "node table: one row per clip × 11 groups")
    au_up = next(c for c in data["node_table"][0]["cells"] if c["group"] == "au_upper")
    check(au_up["med_z"] is None, "group with no columns present → empty cell, not fake 0")
    clip2 = data["clips"][2]
    check(any(t["ch"] == "AU12_velocity_max" and t["z"] > 2 for f in clip2["flags"]
              for t in f["top"]),
          "injected leakage channel surfaces in flagged-window top contributors")

    print("2. dead channels & rendering hygiene")
    check("ear_mean" in data["quality"]["dead_channels"], "all-NaN channel listed as dead")
    check("silent_speech_duration_ms" in data["meta"]["baseline"]["uncalibratable"],
          "null-std baseline channel listed as uncalibratable")
    html_path = generate_report(rec, generated_on="2026-07-10")
    html = open(html_path).read()
    check(os.path.getsize(html_path) > 20000, "report HTML written and non-trivial")
    check("http://" not in html and "https://" not in html,
          "self-contained: zero external URLs (air-gap safe)")
    check("NaN" not in html, "no NaN token leaks into the embedded JSON/JS")
    check("no usable data" in html, "dead channel rendered as an explicit no-data row, not a zero trace")
    check("✓ Baseline healthy" in html, "baseline-health check passes on the healthy fixture")
    check("coupling fit healthy" in html and "data-kind='lane'" in html,
          "healthy coupling fit → lane rendered")
    check("VALIDATION MODE" not in html and "annotated Truth" not in html,
          "no ground-truth overlay by default (production mode)")

print("3. degenerate baseline fires the loud alert")
with tempfile.TemporaryDirectory() as tmp:
    rec, _ = make_fixture(tmp, degenerate_baseline=True, coupling="none")
    html = open(generate_report(rec)).read()
    check("DEGENERATE BASELINE" in html, "near-zero baseline deviation → degenerate alert")

print("4. coupling lane conditionality")
with tempfile.TemporaryDirectory() as tmp:
    rec, _ = make_fixture(tmp, coupling="degenerate")
    data = build_report_data(rec)
    html = open(generate_report(rec)).read()
    check(all(c["coupling"] is None for c in data["clips"]),
          "degenerate fit (ratio ≥ 0.90) → coupling data withheld")
    check("DEGENERATE" in data["meta"]["coupling_status"].upper() and "data-kind='lane'" not in html,
          "report states WHY the lane is withheld")
with tempfile.TemporaryDirectory() as tmp:
    rec, _ = make_fixture(tmp, coupling="none")
    html = open(generate_report(rec)).read()
    check("coupling attribution not run" in html, "missing attribution file → 'not run' note")

print("5. ELAN overlay is strictly validation-mode")
with tempfile.TemporaryDirectory() as tmp:
    rec, _ = make_fixture(tmp)
    elan_dir = os.path.join(tmp, "elan"); os.makedirs(elan_dir)
    open(os.path.join(elan_dir, "B04C002_test.eaf"), "w").write(EAF)   # → file_index 1
    data = build_report_data(rec, elan_dir=elan_dir)
    check(data["clips"][1]["elan"] == [["Truth", 5000, 15000], ["Lie", 30000, 45000]],
          "eaf parsed onto the matching clip (C002 → file_index 1)")
    check(data["clips"][2]["elan"] is None, "clips without an eaf stay unannotated")
    html = open(generate_report(rec, elan_dir=elan_dir)).read()
    check("VALIDATION MODE" in html and "annotated Truth" in html,
          "validation mode is loudly labeled in the report header + legend")

print(f"\nverify_report: {ok} checks passed — no GPU, no real footage, no network.")
