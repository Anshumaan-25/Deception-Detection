"""Produce a scorecard-compatible ELAN dir for SubjectB WITHOUT touching the
pre-registered replication_scorecard.py (which maps interview file_index f -> *C{f+1:03d}*.eaf).

SubjectB's raw eafs carry two colliding original-ID series (B06Cxxx, B41Cxxx), so the
C###-glob mis-pairs them. Here we copy each interview's *verified* eaf (subjectB_manifest.json)
to a unique C{f+1:03d} name matching its true file_index, applying label_normalizations
(fixes the 'LIe' typo -> 'Lie') to the annotation text. Baseline (fidx 0) is not scored, so
it is not copied. Output: pipeline_system_outputs/SUBJECTB_SRC/elan_normalized/.

Run: cwd=deception_detection, any python. Idempotent (overwrites)."""
import json, os
import xml.etree.ElementTree as ET

MAN = "validation/gt_subjectB/subjectB_manifest.json"
man = json.load(open(MAN))
src_elan = man["elan_dir"]
norm = {k: v for k, v in man.get("label_normalizations", {}).items()}
outdir = "pipeline_system_outputs/SUBJECTB_SRC/elan_normalized"
os.makedirs(outdir, exist_ok=True)

written = []
for c in man["clips"]:
    if c["role"] != "interview":
        continue
    fidx = c["file_index"]
    src = os.path.join(src_elan, c["eaf"])
    tree = ET.parse(src)
    n_fixed = 0
    for av in tree.getroot().iter("ANNOTATION_VALUE"):
        if av.text in norm:
            av.text = norm[av.text]; n_fixed += 1
    out = os.path.join(outdir, f"C{fidx+1:03d}_subjectB.eaf")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    written.append((fidx, c["eaf"], os.path.basename(out), n_fixed))
    print(f"  fidx {fidx}: {c['eaf']:28s} -> {os.path.basename(out):20s} "
          f"(labels normalized: {n_fixed})")
print(f"[normalize_elan] wrote {len(written)} interview eafs to {outdir}")
