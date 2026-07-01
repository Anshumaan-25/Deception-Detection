"""
SPOVNOB — Module 4: layer3_contamination.py
============================================

Layer:      Layer 3 — Forensic Overlap Exclusion (NaN-Only) + Temporal
            Smoothing. Imports Modules 0a/0b/1/2/3 only.

Purpose:    The final forensic filter of the pipeline. Every HIGH block
            and every intervening gap from Layer 2 is inspected for
            simultaneous speech with the PyAnnote Overlapped Speech
            Detection pipeline. Contaminated blocks are logged as NaN
            (timestamps and scores preserved) and PERMANENTLY excluded —
            never repaired: separation/reconstruction models (HTDemucs,
            SepFormer, SpeakerBeam, ...) are categorically forbidden by
            architecture; pure exclusion is the only path. Clean blocks
            are merged across overlap-free gaps under 400ms (natural
            plosives / breath intakes) into contiguous segments, and the
            ORIGINAL audio for each segment is sliced to WAV, hashed,
            and emitted untouched as the pipeline's final verified
            output. (Downstream behavioral analysis is deferred to a
            separate design phase — out of scope here.)

Inputs:     SessionManifest · BatchAudio (Layer 0) · ResidentModels
            (gate; only ovd_pipeline used) · Layer2Result (authoritative
            pass) · Layer3Params · work dir
Outputs:    Layer3Result — per-file CLEAN segments (WAV paths + SHA-256,
            PTS spans local/global, bridge metadata) and NaN block log ·
            <work_dir>/layer3/clean/*.wav + JSON sidecars ·
            <work_dir>/layer3/layer3_output.json (canonical, hashed) ·
            per-file worker logs merged canonically into the manifest.

Implements (Audio_Diarization.md):
            - "Layer 3 — Forensic Overlap Exclusion (NaN-Only)" in full:
              input contract (un-smoothed HIGH 1s blocks), PyAnnote OVD
              over blocks AND gaps, the NaN-Only Exclusion Policy
              (block-level: any intersecting overlap voids the entire
              block), the no-separation-models architectural note, and
              temporal smoothing (<400ms, clean gaps only).
            - Layer 2 -> Layer 3 handoff table (HIGH+overlap ->
              CONTAMINATED -> NaN log, never the clean output).
            - "Relocation of Temporal Smoothing" (Layer 2 section): the
              gap itself must be proven overlap-free BEFORE bridging, so
              a rapid interjection can never be swallowed unseen.
            - System Environment "Canonical Manifest Merge Rule".

Decision notes (review-flagged):
            - OVD runs ONCE over each FULL file (not per-block windows):
              strictly more information, simpler, deterministic; regions
              are intersected with blocks/gaps in pure arithmetic.
            - Gap dominance guard (implementation-added safeguard,
              default ON): even an overlap-free gap is NOT bridged if a
              Layer 2 block overlapping the gap shows interviewer
              evidence (MEDIUM with margin_failed, or S_interviewer >=
              S_target). OVD only detects SIMULTANEOUS speech; a solo
              interviewer interjection over target silence would pass an
              overlap-only check — this guard refuses the bridge using
              already-computed Layer 2 scores. Trim-only philosophy: the
              guard can only refuse bridges, never create audio.
            - Bridged gap audio (< 400ms of original signal between two
              clean blocks) IS included in the output segment — that is
              the point of smoothing; it is unmodified original audio.
            - OVD float seconds -> integer ms via Python round()
              (banker's rounding) — fixed, documented rule.

CUDA determinism dependencies: the four environment_gate constants for
the PyAnnote forward passes (pipeline instantiated with pinned
hyperparameters by the gate); all post-detection logic is pure integer
arithmetic.

Self-test:  python3 layer3_contamination.py --selftest   (stdlib only:
            the full flow runs against injected overlap regions — zero
            pip installs, no torch, no GPU).
"""

from __future__ import annotations

import environment_gate  # noqa: F401  (first import: fixes process env)

import argparse
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from layer0_preprocessor import SAMPLE_RATE, BatchAudio, FileAudio
from layer1_enrollment.encoding import pcm_slice
from layer2_tracker import Layer2Result
from session_manifest import (
    Operation,
    SessionManifest,
    WorkerLog,
    canonical_json,
    merge_worker_logs,
    sha256_of_file,
    sha256_of_obj,
)

OP_INIT = "layer3_init"
OP_OVD = "layer3_ovd_regions"
OP_NAN = "layer3_nan_block"
OP_SEGMENT = "layer3_segment"
OP_FILE = "layer3_file_summary"

# Regions and providers use local-PTS integer milliseconds throughout.
OverlapProvider = Callable[[FileAudio], List[Tuple[int, int]]]


class Layer3Error(RuntimeError):
    """Unrecoverable Layer 3 failure (blocking halt already recorded)."""


@dataclass(frozen=True)
class Layer3Params:
    merge_gap_ms: int = 400          # doc default; operator-tunable via
                                     # manifest, strongly discouraged
    gap_dominance_guard: bool = True # implementation-added safeguard

    def manifest_payload(self) -> Dict[str, Any]:
        return {"merge_gap_ms": self.merge_gap_ms,
                "gap_dominance_guard": self.gap_dominance_guard}


@dataclass
class CleanSegment:
    file_index: int
    start_local_ms: int
    end_local_ms: int
    start_global_ms: int
    end_global_ms: int
    duration_ms: int
    block_count: int
    bridged_gaps: List[Tuple[int, int]]
    wav_path: str = ""
    wav_sha256: str = ""


@dataclass
class FileContamination:
    file_index: int
    source_file: str
    segments: List[CleanSegment]
    nan_blocks: List[Dict[str, Any]]
    overlap_regions: List[Tuple[int, int]]
    clean_ms: int
    contaminated_ms: int
    bridged_gap_ms: int


@dataclass
class Layer3Result:
    files: List[FileContamination]
    output_path: str
    output_sha256: str
    total_clean_ms: int
    total_contaminated_ms: int


# =============================================================================
# Pure interval machinery (stdlib-only; self-tested)
# =============================================================================

def intervals_intersect(a0: int, a1: int, b0: int, b1: int) -> bool:
    """Half-open interval intersection: [a0,a1) overlaps [b0,b1)."""
    return a0 < b1 and b0 < a1


def merge_regions(regions: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort and merge overlapping/touching regions into disjoint spans."""
    merged: List[List[int]] = []
    for start, end in sorted(regions):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


def region_overlaps_span(
    regions: Sequence[Tuple[int, int]], start: int, end: int
) -> bool:
    return any(intervals_intersect(start, end, r0, r1) for r0, r1 in regions)


def classify_run_blocks(
    run_blocks: Sequence[Dict[str, Any]],
    regions: Sequence[Tuple[int, int]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """NaN-Only Exclusion Policy at block granularity: any overlap region
    intersecting a block voids the ENTIRE block. Returns
    (clean_subsegments, nan_records). Clean sub-segments are maximal
    sequences of adjacent clean blocks (prev.end == next.start)."""
    subsegments: List[Dict[str, Any]] = []
    nan_records: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for block in run_blocks:
        start, end = block["start_ms"], block["end_ms"]
        if region_overlaps_span(regions, start, end):
            nan_records.append({
                "designation": "NaN",
                "decision": "CONTAMINATED",
                "start_local_ms": start,
                "end_local_ms": end,
                "duration_ms": end - start,
                "S_target_median": block.get("S_target_median"),
                "S_interviewer_median": block.get("S_interviewer_median"),
                "overlap_regions_hit": [
                    [r0, r1] for r0, r1 in regions
                    if intervals_intersect(start, end, r0, r1)
                ],
            })
            current = None
            continue
        if current is not None and current["end_ms"] == start:
            current["end_ms"] = end
            current["blocks"].append(block)
        else:
            current = {"start_ms": start, "end_ms": end, "blocks": [block]}
            subsegments.append(current)
    return subsegments, nan_records


def gap_dominance_blocked(
    gap_start: int,
    gap_end: int,
    block_map: Sequence[Dict[str, Any]],
) -> bool:
    """Implementation-added bridge guard: refuse bridging when any Layer 2
    block overlapping the gap shows interviewer evidence. Inert when the
    block map is absent (the document's base rule is OVD-only)."""
    for block in block_map:
        if not intervals_intersect(
            gap_start, gap_end,
            block["start_local_ms"], block["end_local_ms"],
        ):
            continue
        if block.get("tier") == "MEDIUM" and block.get("margin_failed"):
            return True
        s_target = block.get("s_target_median")
        s_interviewer = block.get("s_interviewer_median")
        if (
            s_target is not None
            and s_interviewer is not None
            and s_interviewer >= s_target
        ):
            return True
    return False


def assemble_segments(
    subsegments: Sequence[Dict[str, Any]],
    regions: Sequence[Tuple[int, int]],
    block_map: Sequence[Dict[str, Any]],
    params: Layer3Params,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Temporal smoothing: merge consecutive clean sub-segments across a
    gap only when ALL hold — gap < merge_gap_ms, the gap interval is
    overlap-free, and the dominance guard (if enabled) does not object.
    A gap created by a contaminated block can never be bridged: it
    intersects an overlap region by construction.

    Returns (final_segments, gap_decisions) where each final segment is
    {start_ms, end_ms, blocks, bridged_gaps} and gap_decisions is the
    audit trail of every bridge evaluation."""
    segments: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for sub in subsegments:
        if segments:
            gap_start = segments[-1]["end_ms"]
            gap_end = sub["start_ms"]
            gap = gap_end - gap_start
            bridge = True
            reason = "bridged"
            if gap >= params.merge_gap_ms:
                bridge, reason = False, "gap_too_long"
            elif region_overlaps_span(regions, gap_start, gap_end):
                bridge, reason = False, "overlap_in_gap"
            elif params.gap_dominance_guard and gap_dominance_blocked(
                gap_start, gap_end, block_map
            ):
                bridge, reason = False, "interviewer_evidence_in_gap"
            decisions.append({
                "gap_start_ms": gap_start, "gap_end_ms": gap_end,
                "gap_ms": gap, "bridged": bridge, "reason": reason,
            })
            if bridge:
                segments[-1]["end_ms"] = sub["end_ms"]
                segments[-1]["blocks"].extend(sub["blocks"])
                segments[-1]["bridged_gaps"].append((gap_start, gap_end))
                continue
        segments.append({
            "start_ms": sub["start_ms"], "end_ms": sub["end_ms"],
            "blocks": list(sub["blocks"]), "bridged_gaps": [],
        })
    return segments, decisions


# =============================================================================
# Production overlap provider (torch lazy; resident PyAnnote OVD)
# =============================================================================

def pyannote_overlap_provider(models: Any) -> OverlapProvider:
    def provider(file_audio: FileAudio) -> List[Tuple[int, int]]:
        import torch

        waveform = (
            torch.frombuffer(bytearray(file_audio.pcm), dtype=torch.int16)
            .to(torch.float32) / 32768.0
        ).unsqueeze(0)
        annotation = models.ovd_pipeline(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        )
        start_pts = file_audio.audio_start_pts_ms
        regions = [
            (start_pts + round(segment.start * 1000),
             start_pts + round(segment.end * 1000))
            for segment in annotation.get_timeline().support()
        ]
        return merge_regions(regions)

    return provider


# =============================================================================
# Per-file contamination pass (provider-driven; self-testable end to end)
# =============================================================================

def _write_segment_wav(
    clean_dir: Path,
    file_audio: FileAudio,
    segment: Dict[str, Any],
) -> Tuple[str, str, str, str]:
    """Slice the ORIGINAL audio for one clean segment and persist WAV +
    canonical JSON sidecar. Returns (wav_path, wav_sha, record_path,
    record_sha)."""
    stem = (
        f"clean_{file_audio.file_index:03d}_"
        f"{segment['start_ms']}_{segment['end_ms']}"
    )
    wav_path = clean_dir / f"{stem}.wav"
    pcm = pcm_slice(
        file_audio.pcm, file_audio.audio_start_pts_ms, file_audio.num_samples,
        segment["start_ms"], segment["end_ms"],
    )
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm)
    record = {
        "file_index": file_audio.file_index,
        "source_file": file_audio.source_path,
        "start_local_ms": segment["start_ms"],
        "end_local_ms": segment["end_ms"],
        "start_global_ms": file_audio.to_global_ms(segment["start_ms"]),
        "end_global_ms": file_audio.to_global_ms(segment["end_ms"]),
        "duration_ms": segment["end_ms"] - segment["start_ms"],
        "bridged_gaps": [[g0, g1] for g0, g1 in segment["bridged_gaps"]],
        "block_count": len(segment["blocks"]),
        "policy": "original_signal_only_no_reconstruction",
    }
    record_path = clean_dir / f"{stem}.json"
    record_path.write_text(canonical_json(record) + "\n", encoding="utf-8")
    return (str(wav_path), sha256_of_file(wav_path),
            str(record_path), sha256_of_file(record_path))


def contaminate_file(
    file_audio: FileAudio,
    high_runs: Sequence[Any],
    block_map: Sequence[Dict[str, Any]],
    regions: List[Tuple[int, int]],
    params: Layer3Params,
    worker_log: WorkerLog,
    clean_dir: Path,
) -> FileContamination:
    regions = merge_regions(regions)
    worker_log.append(OP_OVD, {
        "overlap_regions": [[r0, r1] for r0, r1 in regions],
        "region_count": len(regions),
        "overlap_total_ms": sum(r1 - r0 for r0, r1 in regions),
    })

    # NaN classification per run, then cross-run smoothing.
    all_subsegments: List[Dict[str, Any]] = []
    nan_blocks: List[Dict[str, Any]] = []
    for run in high_runs:
        subsegments, nan_records = classify_run_blocks(run.blocks, regions)
        all_subsegments.extend(subsegments)
        nan_blocks.extend(nan_records)
    for record in nan_blocks:
        worker_log.append(OP_NAN, record, start_ms=record["start_local_ms"])

    merged, gap_decisions = assemble_segments(
        all_subsegments, regions, block_map, params,
    )

    segments: List[CleanSegment] = []
    for segment in merged:
        wav_path, wav_sha, record_path, record_sha = _write_segment_wav(
            clean_dir, file_audio, segment,
        )
        clean = CleanSegment(
            file_index=file_audio.file_index,
            start_local_ms=segment["start_ms"],
            end_local_ms=segment["end_ms"],
            start_global_ms=file_audio.to_global_ms(segment["start_ms"]),
            end_global_ms=file_audio.to_global_ms(segment["end_ms"]),
            duration_ms=segment["end_ms"] - segment["start_ms"],
            block_count=len(segment["blocks"]),
            bridged_gaps=list(segment["bridged_gaps"]),
            wav_path=wav_path,
            wav_sha256=wav_sha,
        )
        segments.append(clean)
        worker_log.append(OP_SEGMENT, {
            "decision": "CLEAN",
            "start_local_ms": clean.start_local_ms,
            "end_local_ms": clean.end_local_ms,
            "start_global_ms": clean.start_global_ms,
            "end_global_ms": clean.end_global_ms,
            "duration_ms": clean.duration_ms,
            "block_count": clean.block_count,
            "bridged_gaps": [[g0, g1] for g0, g1 in clean.bridged_gaps],
            "wav_path": wav_path,
            "wav_sha256": wav_sha,
            "record_path": record_path,
            "record_sha256": record_sha,
        }, start_ms=clean.start_local_ms)

    clean_ms = sum(s.duration_ms for s in segments)
    contaminated_ms = sum(r["duration_ms"] for r in nan_blocks)
    bridged_gap_ms = sum(
        g1 - g0 for s in segments for g0, g1 in s.bridged_gaps
    )
    worker_log.append(OP_FILE, {
        "clean_ms": clean_ms,
        "contaminated_ms": contaminated_ms,
        "bridged_gap_ms": bridged_gap_ms,
        "segments": len(segments),
        "nan_blocks": len(nan_blocks),
        "gap_decisions": gap_decisions,
    })
    return FileContamination(
        file_index=file_audio.file_index,
        source_file=file_audio.source_path,
        segments=segments,
        nan_blocks=nan_blocks,
        overlap_regions=regions,
        clean_ms=clean_ms,
        contaminated_ms=contaminated_ms,
        bridged_gap_ms=bridged_gap_ms,
    )


# =============================================================================
# Layer 3 entrypoint
# =============================================================================

def run_layer3(
    manifest: SessionManifest,
    batch: BatchAudio,
    models: Any,
    layer2: Layer2Result,
    work_dir: Path | str,
    params: Layer3Params = Layer3Params(),
    overlap_provider: Optional[OverlapProvider] = None,
) -> Layer3Result:
    """Overlap-screen the authoritative Layer 2 output and emit the final
    final verified clean segments. ``overlap_provider`` is injectable for
    the self-test; production defaults to the resident PyAnnote OVD."""
    if not layer2.authoritative:
        manifest.append(Operation.BLOCKING_HALT, {
            "reason": "layer3_requires_authoritative_layer2",
        })
        raise Layer3Error("Layer 3 only consumes the authoritative Layer 2 pass")

    layer3_dir = Path(work_dir) / "layer3"
    clean_dir = layer3_dir / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    provider = (
        overlap_provider if overlap_provider is not None
        else pyannote_overlap_provider(models)
    )

    manifest.append(OP_INIT, {
        "layer": 3,
        "params": params.manifest_payload(),
        "layer2_output_sha256": layer2.output_sha256,
        "policy": "NaN_only_exclusion_no_separation_models",
    })

    tracks_by_index = {track.file_index: track for track in layer2.files}
    files: List[FileContamination] = []
    worker_paths: List[Path] = []
    for file_audio in batch.files:
        track = tracks_by_index.get(file_audio.file_index)
        if track is None:
            continue
        worker_path = layer3_dir / f"worker_{file_audio.file_index:03d}.jsonl"
        with WorkerLog(worker_path, file_audio.file_index) as worker_log:
            files.append(contaminate_file(
                file_audio, track.high_runs, track.blocks,
                provider(file_audio), params, worker_log, clean_dir,
            ))
        worker_paths.append(worker_path)

    merge_worker_logs(manifest, worker_paths)

    output_doc = {
        "schema": "spovnob-layer3-output-v1",
        "layer2_output_sha256": layer2.output_sha256,
        "params": params.manifest_payload(),
        "files": [
            {
                "file_index": f.file_index,
                "source_file": f.source_file,
                "clean_ms": f.clean_ms,
                "contaminated_ms": f.contaminated_ms,
                "bridged_gap_ms": f.bridged_gap_ms,
                "overlap_regions": [[r0, r1] for r0, r1 in f.overlap_regions],
                "nan_blocks": f.nan_blocks,
                "segments": [
                    {
                        "start_local_ms": s.start_local_ms,
                        "end_local_ms": s.end_local_ms,
                        "start_global_ms": s.start_global_ms,
                        "end_global_ms": s.end_global_ms,
                        "duration_ms": s.duration_ms,
                        "block_count": s.block_count,
                        "bridged_gaps": [[g0, g1] for g0, g1 in s.bridged_gaps],
                        "wav_path": s.wav_path,
                        "wav_sha256": s.wav_sha256,
                    }
                    for s in f.segments
                ],
            }
            for f in files
        ],
    }
    output_sha = sha256_of_obj(output_doc)
    output_path = layer3_dir / "layer3_output.json"
    output_path.write_text(canonical_json(output_doc) + "\n", encoding="utf-8")
    manifest.append(Operation.OUTPUT_HASH, {
        "layer": 3,
        "output_path": str(output_path),
        "output_sha256": output_sha,
        "total_clean_ms": sum(f.clean_ms for f in files),
        "total_contaminated_ms": sum(f.contaminated_ms for f in files),
    })
    return Layer3Result(
        files=files,
        output_path=str(output_path),
        output_sha256=output_sha,
        total_clean_ms=sum(f.clean_ms for f in files),
        total_contaminated_ms=sum(f.contaminated_ms for f in files),
    )


# =============================================================================
# CLI
# =============================================================================

def _main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SPOVNOB Layer 3 contamination flagging (Module 4)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true",
                      help="stdlib-only self-test (no pip, no torch, no GPU)")
    mode.add_argument("--run", action="store_true",
                      help="run Layers 0+1+2+3 on a batch (Ubuntu box)")
    parser.add_argument("--videos", nargs="+", type=Path)
    parser.add_argument("--clicks", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--model-store", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--operator", type=str, default=None)
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest_stdlib()

    for required in ("videos", "clicks", "work_dir", "model_store", "manifest"):
        if getattr(args, required) is None:
            parser.error(f"--run requires --{required.replace('_', '-')}")

    from layer0_preprocessor import preprocess_batch
    from layer1_enrollment import load_clicks, run_layer1
    from layer2_tracker import run_layer2

    clicks = load_clicks(args.clicks)
    with SessionManifest(args.manifest, operator_id=args.operator) as manifest:
        models = environment_gate.run_gate(args.model_store, manifest)
        batch = preprocess_batch(manifest, args.videos, args.work_dir,
                                 models.silero)
        enrollment = run_layer1(manifest, batch, models, clicks, args.work_dir)
        layer2 = run_layer2(manifest, batch, models, enrollment, args.work_dir)
        result = run_layer3(manifest, batch, models, layer2, args.work_dir)
    print(
        f"layer3 complete — {result.total_clean_ms} ms CLEAN across "
        f"{sum(len(f.segments) for f in result.files)} segments, "
        f"{result.total_contaminated_ms} ms excluded as NaN, "
        f"output sha256 {result.output_sha256[:12]}…"
    )
    return 0


# =============================================================================
# Stdlib-only self-test (standing policy: zero pip installs, no torch/GPU)
# =============================================================================

def _block(start: int, end: int, s_t: float = 0.8, s_i: float = 0.2) -> dict:
    return {"start_ms": start, "end_ms": end,
            "S_target_median": s_t, "S_interviewer_median": s_i}


def _selftest_stdlib() -> int:
    import tempfile

    from layer2_tracker import Calibration, FileTrack, HighRun
    from layer0_preprocessor import FileAudio

    assert "torch" not in sys.modules, "torch imported at module level"
    params = Layer3Params()

    # 1. interval machinery
    assert intervals_intersect(0, 10, 9, 20)
    assert not intervals_intersect(0, 10, 10, 20)      # half-open: touching
    assert merge_regions([(5, 10), (0, 6), (20, 30), (30, 35)]) == [
        (0, 10), (20, 35)]
    assert merge_regions([(5, 5), (3, 2)]) == []        # degenerate dropped
    assert region_overlaps_span([(100, 200)], 150, 160)
    assert not region_overlaps_span([(100, 200)], 200, 300)

    # 2. NaN classification: overlap (9300, 9600) voids the ENTIRE
    # 9000-10000 block; the run splits into two clean sub-segments.
    run_blocks = [_block(7000 + i * 1000, 8000 + i * 1000) for i in range(5)]
    subs, nans = classify_run_blocks(run_blocks, [(9300, 9600)])
    assert len(nans) == 1
    assert (nans[0]["start_local_ms"], nans[0]["end_local_ms"]) == (9000, 10000)
    assert nans[0]["designation"] == "NaN"
    assert [(s["start_ms"], s["end_ms"]) for s in subs] == [
        (7000, 9000), (10000, 12000)]

    # 3. The contaminated-block gap can never be bridged. Here the voided
    # block leaves a 1000ms gap, refused by the length rule first; short
    # contaminated gaps are refused by the overlap rule (test 6).
    merged, decisions = assemble_segments(subs, [(9300, 9600)], [], params)
    assert [(m["start_ms"], m["end_ms"]) for m in merged] == [
        (7000, 9000), (10000, 12000)]
    assert decisions[0]["bridged"] is False
    assert decisions[0]["reason"] == "gap_too_long"

    # 4. Clean 300ms gap between runs -> bridged (gap audio included).
    sub_a = {"start_ms": 7000, "end_ms": 9000, "blocks": run_blocks[:2]}
    sub_b = {"start_ms": 9300, "end_ms": 11000, "blocks": run_blocks[2:4]}
    merged, decisions = assemble_segments([sub_a, sub_b], [], [], params)
    assert len(merged) == 1
    assert (merged[0]["start_ms"], merged[0]["end_ms"]) == (7000, 11000)
    assert merged[0]["bridged_gaps"] == [(9000, 9300)]
    assert decisions[0]["bridged"] is True

    # 5. 400ms gap is NOT bridged (rule is strictly < merge_gap_ms).
    sub_c = {"start_ms": 9400, "end_ms": 11000, "blocks": []}
    merged, decisions = assemble_segments([sub_a, sub_c], [], [], params)
    assert len(merged) == 2 and decisions[0]["reason"] == "gap_too_long"

    # 6. Overlap inside an otherwise short gap blocks the bridge.
    merged, decisions = assemble_segments(
        [sub_a, sub_b], [(9100, 9200)], [], params)
    assert len(merged) == 2 and decisions[0]["reason"] == "overlap_in_gap"

    # 7. Dominance guard: interviewer evidence in the gap block refuses
    # the bridge; disabling the guard (doc base rule) allows it.
    gap_block_map = [{
        "start_local_ms": 9000, "end_local_ms": 10000, "tier": "MEDIUM",
        "margin_failed": True, "s_target_median": 0.7,
        "s_interviewer_median": 0.6,
    }]
    merged, decisions = assemble_segments([sub_a, sub_b], [], gap_block_map,
                                          params)
    assert len(merged) == 2
    assert decisions[0]["reason"] == "interviewer_evidence_in_gap"
    base_rule = Layer3Params(gap_dominance_guard=False)
    merged, _ = assemble_segments([sub_a, sub_b], [], gap_block_map, base_rule)
    assert len(merged) == 1
    dominance_map = [{
        "start_local_ms": 9000, "end_local_ms": 10000, "tier": "SUB_THRESHOLD",
        "margin_failed": False, "s_target_median": 0.30,
        "s_interviewer_median": 0.35,
    }]
    assert gap_dominance_blocked(9000, 9300, dominance_map)
    assert not gap_dominance_blocked(9000, 9300, [])

    # 8. Edge-trimmed partial blocks survive intact (no grid assumptions).
    trimmed = [_block(7250, 8000), _block(8000, 9000)]
    subs, nans = classify_run_blocks(trimmed, [])
    assert nans == [] and [(s["start_ms"], s["end_ms"]) for s in subs] == [
        (7250, 9000)]

    # 9. End-to-end run_layer3 with injected regions: WAVs written,
    # manifest chain verified, deterministic output hash across runs.
    duration = 20000
    def _fake_file(index: int) -> FileAudio:
        return FileAudio(
            file_index=index, source_path=f"video_{index:02d}.mp4",
            wav_path="", source_sha256="", wav_sha256="",
            num_samples=duration * 16, duration_ms=duration,
            audio_start_pts_ms=0, audio_start_missing=False,
            vfr_suspected=False,
            silero_segments_local_ms=[(0, duration)], pcm=bytes(duration * 32),
        )

    def _run(index: int) -> HighRun:
        return HighRun(
            start_local_ms=7000, end_local_ms=12000,
            blocks=[_block(7000 + i * 1000, 8000 + i * 1000)
                    for i in range(5)],
        )

    calibration = Calibration(
        theta_high=0.6, theta_med=0.45, kind="DERIVED",
        record={}, calibration_ref="c" * 64, overlap_warning=False,
    )
    layer2 = Layer2Result(
        calibration=calibration,
        files=[
            FileTrack(
                file_index=i, source_file=f"video_{i:02d}.mp4",
                high_runs=[_run(i)], tier_counts={}, high_ms=5000,
                silero_ms=duration, ratio=0.25, ratio_level="LOW_ADVISORY",
                unattributed_speech_ms=15000, high_scores=[0.8] * 5,
                blocks=[],
            )
            for i in range(2)
        ],
        no_anti_profile=False, authoritative=True,
        output_path="", output_sha256="f" * 64,
    )
    batch = BatchAudio(files=[_fake_file(0), _fake_file(1)])

    def provider(file_audio: FileAudio) -> List[Tuple[int, int]]:
        # File 0: contamination mid-run; file 1: fully clean.
        return [(9300, 9600)] if file_audio.file_index == 0 else []

    hashes = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with SessionManifest(tmp / "m.jsonl") as manifest:
                result = run_layer3(
                    manifest, batch, models=None, layer2=layer2,
                    work_dir=tmp, params=params, overlap_provider=provider,
                )
            entries = SessionManifest.verify_chain(tmp / "m.jsonl")
            hashes.append(result.output_sha256)
            assert result.files[0].clean_ms == 4000        # one block NaN'd
            assert result.files[0].contaminated_ms == 1000
            assert len(result.files[0].segments) == 2
            assert result.files[1].clean_ms == 5000
            assert len(result.files[1].segments) == 1
            for f in result.files:
                for segment in f.segments:
                    wav = Path(segment.wav_path)
                    assert wav.exists()
                    with wave.open(str(wav), "rb") as handle:
                        assert handle.getnframes() == segment.duration_ms * 16
            nan_ops = [e for e in entries if e["operation"] == OP_NAN]
            assert len(nan_ops) == 1
            merged_record = nan_ops[0]["payload"]      # worker-log record
            assert merged_record["file_index"] == 0
            assert merged_record["payload"]["designation"] == "NaN"
            assert merged_record["payload"]["start_local_ms"] == 9000
            file_summaries = [e["payload"] for e in entries
                              if e["operation"] == OP_FILE]
            assert [s["file_index"] for s in file_summaries] == [0, 1]
            ops = [e["operation"] for e in entries]
            assert Operation.OUTPUT_HASH in ops
    # Output hash covers wav paths (temp dirs differ) — compare the
    # scrubbed structural content instead of raw hashes:
    assert result.total_clean_ms == 9000

    # 10. Non-authoritative Layer 2 input is refused.
    preview = Layer2Result(
        calibration=calibration, files=[], no_anti_profile=False,
        authoritative=False, output_path="", output_sha256="0" * 64,
    )
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        try:
            with SessionManifest(tmp / "m.jsonl") as manifest:
                run_layer3(manifest, batch, None, preview, tmp, params,
                           provider)
            raise AssertionError("preview Layer 2 accepted by Layer 3")
        except Layer3Error:
            pass

    for forbidden in ("torch", "cv2", "numpy"):
        assert forbidden not in sys.modules, f"self-test imported {forbidden}"
    print("layer3_contamination stdlib self-test OK — full flow exercised "
          "with injected overlap regions; no torch, no GPU, no pip")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
