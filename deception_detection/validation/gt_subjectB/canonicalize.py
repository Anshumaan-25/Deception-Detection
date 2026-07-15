"""Canonicalize SubjectB's 7 raw MPEG-2 clips to bare SubjectA-style base names
(00_baseline .. 06_interview) so the whole downstream pipeline is convention-identical.
Uses the A/V-sync canonicalizer (auto-trims the 80 ms open-GOP audio-lead).

MUST run with SYSTEM ffmpeg on PATH (/usr/bin, has h264_nvenc) — conda's ffmpeg 4.2.2
lacks NVENC. Invoke with PATH=/usr/bin:$PATH and cwd=deception_detection.
"""
import json, os, sys, time
sys.path.insert(0, os.path.abspath("ffmpeg_ingestion/core"))
from canonicalizer import VideoCanonicalizer

MAN = "validation/gt_subjectB/subjectB_manifest.json"
man = json.load(open(MAN))
src_dir = man["source_dir"]
out_dir = os.path.abspath(man["canonical_dir"])
os.makedirs(out_dir, exist_ok=True)

canon = VideoCanonicalizer()
clips = sys.argv[1:] or [c["base"] for c in man["clips"]]   # optional shard by base
by_base = {c["base"]: c for c in man["clips"]}
done = []
for base in clips:
    c = by_base[base]
    inp = os.path.join(src_dir, c["src_video"])
    vout = os.path.join(out_dir, f"{base}_canonical.mp4")
    if os.path.exists(vout):
        print(f"[skip {base}: already canonical]", flush=True); done.append(base); continue
    t = time.time()
    print(f"\n=== CANON {base}  <- {c['src_video']}  (fidx {c['file_index']}) START ===", flush=True)
    try:
        v, a = canon.process(inp, out_dir, base)
        print(f"=== CANON {base} DONE {time.time()-t:.1f}s -> {os.path.basename(v)} + {os.path.basename(a)} ===", flush=True)
        done.append(base)
    except Exception as e:
        print(f"=== CANON {base} FAIL {time.time()-t:.1f}s: {e} ===", flush=True)
print(f"\n### CANON SHARD DONE: {len(done)}/{len(clips)} -> {done} ###", flush=True)
