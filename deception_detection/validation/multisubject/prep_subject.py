"""Generic subject prep for the N>1 corpus (single-series C-token naming, as
SubjectC..F). For one subject package it produces everything the production run
needs EXCEPT the GPU work and the operator click:

  1. manifest.json                         (fidx -> raw video -> eaf, label-normalized)
  2. {TAG}_SRC/spovnob_input/NN_role.mp4   (bare symlinks to the RAW audio-bearing
                                            videos — SPOVNOB Stage-1 / click UI input)
  3. {TAG}_SRC/elan_normalized/C{f+1:03d}_{tag}.eaf
                                            (title-cased labels; unique C-token per
                                             file_index so replication_scorecard's
                                             *C{f+1:03d}* glob resolves 1:1)

Canonicalization (GPU) is separate (canonicalize_generic.py). Convention:
video `...C{n:03d}-{NN}_{role}.mp4` -> file_index NN, original clip token C{n:03d};
eaf matched by that SAME C-token anywhere in its filename. A missing interview eaf
is fine (that clip is cascaded but not scored). Baseline (file_index 0) is not scored.

Usage: python -m validation.multisubject.prep_subject <subject_dir> <TAG> [--tag name]
  e.g. ... my_videos/02SubjectC_session3 REC_SUBJECTC --tag subjectC
"""
import argparse, glob, json, os, re
import xml.etree.ElementTree as ET

CANON = {"truth": "Truth", "lie": "Lie", "neutral": "Neutral"}   # case-insensitive → canonical
SUFFIX = re.compile(r"-(\d{2})_(baseline|interview)", re.IGNORECASE)
CTOK = re.compile(r"C(\d{3})", re.IGNORECASE)
OUT = "pipeline_system_outputs"


def _label_norm(v):
    return CANON.get((v or "").strip().lower(), (v or "").strip())


def prep(subject_dir, tag_id, tag):
    subject_dir = os.path.abspath(subject_dir)
    vids = sorted(v for v in glob.glob(os.path.join(subject_dir, "*.mp4")))
    eafs = glob.glob(os.path.join(subject_dir, "**", "*.eaf"), recursive=True)
    def eaf_for(ctok):                                   # match by C-token, prefer non-duplicate
        hits = sorted(e for e in eafs if CTOK.search(os.path.basename(e))
                      and CTOK.search(os.path.basename(e)).group(1) == ctok)
        return hits[0] if hits else None

    src = f"{OUT}/{tag_id}_SRC"
    spov = f"{src}/spovnob_input"; enorm = f"{src}/elan_normalized"
    os.makedirs(spov, exist_ok=True); os.makedirs(enorm, exist_ok=True)
    clips = []
    for v in vids:
        name = os.path.basename(v)
        ms, mc = SUFFIX.search(name), CTOK.search(name)
        if not ms:
            print(f"  ! skip (no -NN_role suffix): {name}"); continue
        fidx = int(ms.group(1)); role = ms.group(2).lower()
        ctok = mc.group(1) if mc else None
        base = f"{fidx:02d}_{role}"
        # bare symlink to the RAW audio-bearing video (SPOVNOB / click input)
        link = f"{spov}/{base}.mp4"
        if os.path.islink(link) or os.path.exists(link): os.remove(link)
        os.symlink(v, link)
        # normalized eaf (interviews only; title-cased labels; C{fidx+1:03d} token)
        eaf_src = eaf_for(ctok) if ctok else None
        norm_eaf, nfix, labels = None, 0, {}
        if role == "interview" and eaf_src:
            tree = ET.parse(eaf_src)
            for av in tree.getroot().iter("ANNOTATION_VALUE"):
                new = _label_norm(av.text)
                if new != av.text: nfix += 1
                av.text = new; labels[new] = labels.get(new, 0) + 1
            norm_eaf = f"{enorm}/C{fidx+1:03d}_{tag}.eaf"
            tree.write(norm_eaf, encoding="utf-8", xml_declaration=True)
        clips.append({"file_index": fidx, "role": role, "src_video": name, "base": base,
                      "orig_ctoken": f"C{ctok}" if ctok else None,
                      "eaf": os.path.basename(eaf_src) if eaf_src else None,
                      "eaf_normalized": os.path.basename(norm_eaf) if norm_eaf else None,
                      "labels_norm": labels, "labels_fixed": nfix})
    manifest = {"subject": tag, "recording_id": tag_id, "source_dir": subject_dir,
                "canonical_dir": f"{src}/canonical", "spovnob_input": spov,
                "elan_normalized": enorm, "baseline_file_index": 0, "clips": clips}
    mdir = f"validation/gt_{tag}"; os.makedirs(mdir, exist_ok=True)
    open(f"{mdir}/{tag}_manifest.json", "w").write(json.dumps(manifest, indent=2))
    n_int = sum(1 for c in clips if c["role"] == "interview")
    n_eaf = sum(1 for c in clips if c["eaf_normalized"])
    n_fix = sum(c["labels_fixed"] for c in clips)
    print(f"[prep {tag}] {len(clips)} clips ({n_int} interviews, {n_eaf} scoreable), "
          f"{n_fix} labels case-normalized → {mdir}/{tag}_manifest.json")
    for c in clips:
        if c["role"] == "interview" and not c["eaf_normalized"]:
            print(f"  · file_index {c['file_index']} ({c['orig_ctoken']}) has NO eaf — cascaded, not scored")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("subject_dir"); ap.add_argument("tag_id")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    prep(a.subject_dir, a.tag_id, a.tag)
