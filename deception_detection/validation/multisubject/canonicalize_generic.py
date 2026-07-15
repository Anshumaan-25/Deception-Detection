"""Manifest-driven canonicalization to bare names (generic; A/V-sync 80 ms trim).
Reads a subjectX_manifest.json (prep_subject.py output), canonicalizes every clip's
raw src_video → {base}_canonical.mp4 + {base}_hubert.wav in the manifest's canonical_dir.
MUST run with SYSTEM ffmpeg on PATH (/usr/bin, has h264_nvenc). cwd=deception_detection."""
import json, os, sys, time
sys.path.insert(0, os.path.abspath("ffmpeg_ingestion/core"))
from canonicalizer import VideoCanonicalizer

man = json.load(open(sys.argv[1]))
src_dir = man["source_dir"]; out_dir = os.path.abspath(man["canonical_dir"])
os.makedirs(out_dir, exist_ok=True)
canon = VideoCanonicalizer()
bases = sys.argv[2:] or [c["base"] for c in man["clips"]]
by_base = {c["base"]: c for c in man["clips"]}
done = []
for base in bases:
    c = by_base[base]; inp = os.path.join(src_dir, c["src_video"])
    vout = os.path.join(out_dir, f"{base}_canonical.mp4")
    if os.path.exists(vout):
        print(f"[skip {base}]", flush=True); done.append(base); continue
    t = time.time()
    print(f"=== CANON {man['subject']} {base} <- {c['src_video']} START ===", flush=True)
    try:
        canon.process(inp, out_dir, base)
        print(f"=== CANON {man['subject']} {base} DONE {time.time()-t:.1f}s ===", flush=True)
        done.append(base)
    except Exception as e:
        print(f"=== CANON {man['subject']} {base} FAIL: {e} ===", flush=True)
print(f"### {man['subject']} CANON DONE: {len(done)}/{len(bases)} ###", flush=True)
