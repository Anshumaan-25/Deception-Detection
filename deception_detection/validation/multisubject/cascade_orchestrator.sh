#!/usr/bin/env bash
# Overlap cascades with the serial Stage-1 chain: for each subject D/E/F, wait for its
# SPOVNOB pipeline_output.json, then cascade all its clips 4-way (bounded memory), waiting
# for that subject's cascade to finish before starting the next. This runs ALONGSIDE the
# serial Stage-1 driver, so subject-N's cascade overlaps subject-(N+1)'s Stage-1 — filling
# the GPU without colliding two ~21GB Stage-1 memory spikes.
set -u
cd /home/user1/Documents/Deception_Detection/deception_detection
PY=~/anaconda3/envs/spovnob_env/bin/python
SP=~/anaconda3/envs/spovnob_env/lib/python3.10/site-packages
export LD_LIBRARY_PATH="$(ls -d $SP/nvidia/*/lib 2>/dev/null|tr '\n' ':')$SP/torch/lib"
SESS=/home/user1/Documents/Deception_Detection/audio_diarization/session
NW=6   # concurrent clips per subject

for T in D E F; do
  tag=subject$T; RID=REC_SUBJECT$T; M=validation/gt_$tag/${tag}_manifest.json
  SRC=pipeline_system_outputs/${RID}_SRC
  echo "=== [orch] waiting for $tag Stage-1 ($(date +%H:%M)) ==="
  until [ -f "$SESS/rec_$tag/pipeline_output.json" ]; do sleep 30; done
  echo "=== [orch] $tag Stage-1 ready → cascading ($(date +%H:%M)) ==="
  mapfile -t BASES < <($PY -c "import json;[print(c['base']) for c in json.load(open('$M'))['clips']]")
  # launch NW-way: round-robin clips into NW shards
  for ((i=0;i<NW;i++)); do
    shard=(); for ((j=i;j<${#BASES[@]};j+=NW)); do shard+=("${BASES[$j]}"); done
    [ ${#shard[@]} -eq 0 ] && continue
    setsid env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" $PY validation/multisubject/cascade_generic.py "$M" "${shard[@]}" \
      > "$SRC/orch_cascade_s$i.log" 2>&1 &
  done
  # wait for this subject's cascade to finish (all clip windowed CSVs) or procs to end
  nclips=${#BASES[@]}
  while true; do
    done=$(ls pipeline_system_outputs/${RID}_0*/${RID}_0*_windowed_features.csv 2>/dev/null | wc -l)
    [ "$done" -ge "$nclips" ] && { echo "=== [orch] $tag cascade DONE $done/$nclips ($(date +%H:%M)) ==="; break; }
    ps -eo cmd | grep -q "[c]ascade_generic.py .*$M" || { echo "=== [orch] $tag cascade procs ended $done/$nclips ($(date +%H:%M)) ==="; break; }
    sleep 30
  done
  # assemble this subject
  $PY validation/multisubject/assemble_generic.py "$M" > "$SRC/assemble.log" 2>&1
  echo "=== [orch] $tag ASSEMBLED ($(date +%H:%M)) ==="
done
echo "### ORCH DONE: D/E/F cascaded + assembled ###"
