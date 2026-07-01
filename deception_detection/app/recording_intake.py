"""
Recording Intake — clip/wav pairing for batch mode (audio-diarization merge)
============================================================================
Pure helper used by the batch daemon to turn a *recording bucket* (N clips +
their wavs + a shared audio-diarization ``pipeline_output.json``) into an
ordered list of (mp4, wav) pairs, one per diarization ``file_index``.

Matching is by normalized filename stem (tolerant of the deception
canonicalizer's ``_canonical`` / ``_hubert`` suffixes), so a diarization source
named ``NT-clip27.mp4`` pairs with ``NT-clip27_canonical.mp4`` +
``NT-clip27_hubert.wav``.

No GPU / torch — imports only the (stdlib-only) DiarizationBridge.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# The bridge is pure stdlib; make it importable from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio_isolation.core.diarization_bridge import (  # noqa: E402
    DiarizationBridge,
    _normalize_stem,
)


def collect_recording_clips(
    intake_session_dir: Path | str,
    diarization_output_json: Path | str,
) -> List[Tuple[str, str]]:
    """
    Pair each audio-diarization ``file_index`` with its clip (.mp4) and audio
    (.wav) in the recording bucket, ordered by ``file_index``.

    Returns [(mp4_path, wav_path), ...]. Raises FileNotFoundError if any
    ``file_index`` has no matching mp4 or wav — every diarized clip must be
    present in the bucket to be processed (coverage check).
    """
    bucket = Path(intake_session_dir)
    bridge = DiarizationBridge.from_output_json(diarization_output_json)

    mp4s: Dict[str, Path] = {}
    wavs: Dict[str, Path] = {}
    for f in bucket.iterdir():
        if not f.is_file():
            continue
        suffix = f.suffix.lower()
        if suffix == ".mp4":
            mp4s.setdefault(_normalize_stem(f.name), f)
        elif suffix == ".wav":
            wavs.setdefault(_normalize_stem(f.name), f)

    clips: List[Tuple[str, str]] = []
    for fi in bridge.file_indices():
        stem = _normalize_stem(bridge.source_file(fi))
        mp4 = mp4s.get(stem)
        wav = wavs.get(stem)
        if mp4 is None or wav is None:
            missing = "mp4" if mp4 is None else "wav"
            raise FileNotFoundError(
                f"recording bucket {bucket} is missing the {missing} for "
                f"file_index {fi} (stem {stem!r}); "
                f"mp4 stems={sorted(mp4s)}, wav stems={sorted(wavs)}"
            )
        clips.append((str(mp4), str(wav)))
    return clips
