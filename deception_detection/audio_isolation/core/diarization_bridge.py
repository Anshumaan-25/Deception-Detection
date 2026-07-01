"""
Audio-Diarization Bridge — Diarization Adapter (Merge Phase 1)
==============================================================

Consumes the audio-diarization pipeline's output and adapts it to the deception
detection pipeline's diarization seam contract — the ``pyannote_segments`` list
currently mocked at ``main_pipeline.py:384``::

    list[(speaker_id: str, start_ms: int, end_ms: int)]

The audio-diarization pipeline runs in its OWN pinned ``.venv`` as a sealed
subprocess (its environment gate fail-closes on any dependency-pin drift), so
this bridge imports NOTHING from it. It only:

  (a) [optional] INVOKES the audio-diarization runner as a subprocess, and
  (b) PARSES the resulting ``pipeline_output.json`` (on-disk schema
      ``spovnob-pipeline-output-v1``) and adapts per-clip clean-speech segments.

Black-box integration (locked decision #3): the audio-diarization hash-chained
manifest is NOT consumed here — only the segment timeline + clean-WAV references.

Why "TARGET": the audio-diarization pipeline has already anchored the subject's
identity authoritatively (operator click + biometric lock), so every clean
segment belongs to the one verified target. We label it ``"TARGET"`` and feed
that to the existing seam; the downstream ``anchor_target_identity`` then
trivially resolves it.

Time frames (see MERGE_INTEGRATION_PLAN.md §7): the audio-diarization
``*_local_ms`` are in local-PTS ms (include the file's audio start PTS). The
deception consumers work in the canonical WAV / video-frame timebase.
``offset_ms`` is the parametric hook to reconcile the two; default 0 (assume
aligned) until Phase-4 validation measures it.

Pure standard library (json / os / subprocess / shlex / pathlib / dataclasses).
No numpy, torch, or GPU needed to parse and adapt.

Self-test (from deception_detection/):  python tests/verify_diarization_bridge.py
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# The synthetic speaker label handed to the deception seam (see module docstring).
TARGET_SPEAKER_ID = "TARGET"

# The exact on-disk summary schema this bridge understands (literal string as
# emitted by the audio-diarization pipeline's pipeline_output.json).
DIARIZATION_OUTPUT_SCHEMA = "spovnob-pipeline-output-v1"

# Filename suffixes the deception canonicalizer appends; stripped when mapping a
# deception clip back to its audio-diarization ``file_index`` by stem.
CANONICAL_SUFFIXES = ("_canonical", "_hubert", "_16k")

# One adapted segment as fed to the seam.
Segment = Tuple[str, int, int]


class DiarizationBridgeError(Exception):
    """Raised on schema mismatch, an unmappable clip, or a failed diarization run."""


@dataclass(frozen=True)
class CleanSegment:
    """One audio-diarization clean-speech span (verified target, overlap-excluded)."""

    file_index: int
    start_local_ms: int
    end_local_ms: int
    start_global_ms: int
    end_global_ms: int
    duration_ms: int
    wav_path: str
    wav_sha256: str

    @classmethod
    def from_dict(cls, d: dict) -> "CleanSegment":
        return cls(
            file_index=int(d["file_index"]),
            start_local_ms=int(d["start_local_ms"]),
            end_local_ms=int(d["end_local_ms"]),
            start_global_ms=int(d["start_global_ms"]),
            end_global_ms=int(d["end_global_ms"]),
            duration_ms=int(d["duration_ms"]),
            wav_path=str(d["wav_path"]),
            wav_sha256=str(d["wav_sha256"]),
        )


def _normalize_stem(name: str) -> str:
    """Reduce a path/filename to a comparable stem, stripping the deception
    canonicalizer suffixes. ``NT-clip27_canonical.mp4`` -> ``nt-clip27`` ==
    ``NT-clip27.mp4`` -> ``nt-clip27``."""
    stem = Path(str(name)).stem
    lowered = stem.lower()
    changed = True
    while changed:
        changed = False
        for suf in CANONICAL_SUFFIXES:
            if lowered.endswith(suf):
                lowered = lowered[: -len(suf)]
                changed = True
    return lowered


class DiarizationBridge:
    """Parses an audio-diarization ``pipeline_output.json`` and adapts it to the seam.

    Construct via :meth:`from_output_json` (decoupled mode — preferred) or
    :meth:`run_and_load` (the bridge invokes the diarization pipeline itself).
    """

    def __init__(
        self,
        summary: dict,
        *,
        diarization_root: Optional[Path | str] = None,
    ) -> None:
        schema = summary.get("schema")
        if schema != DIARIZATION_OUTPUT_SCHEMA:
            raise DiarizationBridgeError(
                f"unexpected summary schema {schema!r}; "
                f"expected {DIARIZATION_OUTPUT_SCHEMA!r}"
            )
        self.summary: dict = summary
        self.diarization_root: Optional[Path] = (
            Path(diarization_root) if diarization_root is not None else None
        )

        # file_index -> source_file (diarization-side absolute path)
        self._source_by_index: Dict[int, str] = {
            int(f["file_index"]): str(f["source_file"])
            for f in summary.get("files", [])
        }
        # normalized stem -> file_index (for clip mapping)
        self._index_by_stem: Dict[str, int] = {
            _normalize_stem(src): idx
            for idx, src in self._source_by_index.items()
        }

        # file_index -> sorted list[CleanSegment]
        self._segments_by_index: Dict[int, List[CleanSegment]] = {}
        for raw in summary.get("clean_segments", []):
            seg = CleanSegment.from_dict(raw)
            self._segments_by_index.setdefault(seg.file_index, []).append(seg)
        for segs in self._segments_by_index.values():
            segs.sort(key=lambda s: (s.start_local_ms, s.end_local_ms))

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_output_json(
        cls,
        path: Path | str,
        *,
        diarization_root: Optional[Path | str] = None,
    ) -> "DiarizationBridge":
        """Load and parse a ``pipeline_output.json``. If ``diarization_root`` is
        not given, it is inferred by climbing parents from the JSON file until the
        first segment's relative ``wav_path`` resolves to a real file."""
        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)

        if diarization_root is None:
            diarization_root = cls._infer_root(json_path, summary)

        return cls(summary, diarization_root=diarization_root)

    @classmethod
    def run_and_load(
        cls,
        *,
        videos: Sequence[Path | str],
        clicks_path: Path | str,
        work_dir: Path | str,
        model_store: Path | str,
        manifest_path: Path | str,
        diarization_root: Path | str,
        operator: Optional[str] = None,
        env_setup: str = "env.sh",
        timeout: Optional[float] = None,
    ) -> "DiarizationBridge":
        """Invoke the audio-diarization runner as a sealed subprocess (its own
        ``.venv`` via ``source env.sh``), then load the resulting
        ``pipeline_output.json``.

        This is the optional coupled mode (MERGE_INTEGRATION_PLAN.md §6). The
        preferred decoupled flow runs the diarization separately and uses
        :meth:`from_output_json`. NOT exercised by the offline self-test (needs
        the GPU venv + model store)."""
        root = Path(diarization_root)
        args = [
            "python", "pipeline_runner.py", "--run",
            "--videos", *[str(v) for v in videos],
            "--clicks", str(clicks_path),
            "--work-dir", str(work_dir),
            "--model-store", str(model_store),
            "--manifest", str(manifest_path),
        ]
        if operator:
            args += ["--operator", operator]
        inner = f"cd {shlex.quote(str(root))} && source {shlex.quote(env_setup)} && " \
                + " ".join(shlex.quote(a) for a in args)
        proc = subprocess.run(
            ["bash", "-lc", inner],
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise DiarizationBridgeError(
                "audio-diarization run failed (exit "
                f"{proc.returncode}).\nSTDERR:\n{proc.stderr.strip()}"
            )
        output_json = Path(work_dir) / "pipeline_output.json"
        if not output_json.is_absolute():
            output_json = root / output_json
        return cls.from_output_json(output_json, diarization_root=root)

    @staticmethod
    def _infer_root(json_path: Path, summary: dict) -> Optional[Path]:
        segs = summary.get("clean_segments") or []
        if not segs:
            return json_path.resolve().parent
        sample_rel = str(segs[0].get("wav_path", ""))
        if not sample_rel or os.path.isabs(sample_rel):
            return json_path.resolve().parent
        p = json_path.resolve().parent
        for _ in range(8):
            if (p / sample_rel).exists():
                return p
            if p.parent == p:
                break
            p = p.parent
        return json_path.resolve().parent

    # ------------------------------------------------------------------ #
    # Accessors
    # ------------------------------------------------------------------ #
    @property
    def schema(self) -> str:
        return str(self.summary.get("schema"))

    def file_indices(self) -> List[int]:
        """All diarization file indices present in the batch, ascending."""
        return sorted(self._source_by_index)

    def source_file(self, file_index: int) -> str:
        return self._source_by_index[int(file_index)]

    def file_offset_ms(self, file_index: int) -> int:
        """Cumulative offset of this file on the batch's global timeline
        (sum of durations of preceding files in canonical order)."""
        for f in self.summary.get("files", []):
            if int(f["file_index"]) == int(file_index):
                return int(f.get("file_offset_ms", 0))
        raise DiarizationBridgeError(f"unknown file_index {file_index}")

    def clean_segments(self, file_index: int) -> List[CleanSegment]:
        """Raw CleanSegment objects for a file (empty list if none)."""
        return list(self._segments_by_index.get(int(file_index), []))

    def wav_paths_for(self, file_index: int) -> List[Path]:
        """Resolved clean-WAV paths for a file (absolute where resolvable)."""
        return [self._resolve_wav(s.wav_path) for s in self.clean_segments(file_index)]

    def _resolve_wav(self, rel: str) -> Path:
        if os.path.isabs(rel):
            return Path(rel)
        if self.diarization_root is not None:
            return (self.diarization_root / rel)
        return Path(rel)

    def index_for_clip(self, clip: Path | str) -> int:
        """Map a deception clip (original or ``_canonical``/``_hubert``) to its
        diarization ``file_index`` by normalized stem. Raises if unmappable."""
        stem = _normalize_stem(str(clip))
        try:
            return self._index_by_stem[stem]
        except KeyError:
            raise DiarizationBridgeError(
                f"cannot map clip {clip!r} (stem {stem!r}) to a diarization file_index; "
                f"available stems: {sorted(self._index_by_stem)}"
            ) from None

    # ------------------------------------------------------------------ #
    # The seam adapter — the load-bearing output
    # ------------------------------------------------------------------ #
    def segments_for(
        self,
        file_index: int,
        *,
        clock: str = "local",
        offset_ms: int = 0,
    ) -> List[Segment]:
        """Adapt one file's clean segments to the seam contract
        ``[(\"TARGET\", start_ms, end_ms), ...]``.

        clock="local"  → per-clip timeline (use when processing a clip alone).
        clock="global" → batch timeline.
        offset_ms      → subtracted from every boundary to reconcile the
                         diarization local-PTS ms with the consumer's timebase
                         (§7); start is clamped at 0.
        """
        if clock not in ("local", "global"):
            raise DiarizationBridgeError(f"clock must be 'local' or 'global', got {clock!r}")
        out: List[Segment] = []
        for s in self.clean_segments(file_index):
            if clock == "local":
                start, end = s.start_local_ms, s.end_local_ms
            else:
                start, end = s.start_global_ms, s.end_global_ms
            start = max(0, start - offset_ms)
            end = max(start, end - offset_ms)
            out.append((TARGET_SPEAKER_ID, start, end))
        return out

    def segments_for_clip(
        self,
        clip: Path | str,
        *,
        clock: str = "local",
        offset_ms: int = 0,
    ) -> List[Segment]:
        """Convenience: :meth:`index_for_clip` + :meth:`segments_for`."""
        return self.segments_for(
            self.index_for_clip(clip), clock=clock, offset_ms=offset_ms
        )

    # ------------------------------------------------------------------ #
    # Summary roll-ups (diagnostics)
    # ------------------------------------------------------------------ #
    def total_clean_ms(self) -> int:
        return int(self.summary.get("layer3", {}).get("total_clean_ms", 0))

    def segment_count(self) -> int:
        return int(self.summary.get("layer3", {}).get("segment_count", 0))

    def enrollment_quality(self) -> Optional[str]:
        return self.summary.get("enrollment", {}).get("final_quality_state")
