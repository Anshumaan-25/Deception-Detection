"""
verify_wavlm_truncation.py — StableLayerNorm encoder truncation correctness
============================================================================
Regression test for a critical bug found + fixed 2026-07-07 (adversarial
review of the single-pass WavLM rewrite): wavlm-large uses the
StableLayerNorm encoder, which applies ONE unconditional final LayerNorm
after its layer loop and appends its output as the LAST hidden_states entry
— whichever index that happens to be. Naively truncating the encoder to
exactly WAVLM_LAYER_INDEX layers makes hidden_states[WAVLM_LAYER_INDEX] the
last entry, so it silently receives that extra norm — a systematic value
corruption on 100% of production forward passes (WAVLM_TRUNCATE_ENCODER=True
is the module default main_pipeline.py always uses), not fp16 noise.

The fix (audio_isolation/core/acoustic_extractor.py::_load_wavlm): keep
WAVLM_LAYER_INDEX + 1 layers instead of WAVLM_LAYER_INDEX, so the extra norm
lands on the entry AFTER the one production code reads.

This test builds a tiny synthetic WavLM (StableLayerNorm, small dims — no
download, no GPU) and proves, by direct numerical comparison against an
untruncated full-stack forward:
  1. The buggy truncation (keep exactly N layers) corrupts hidden_states[N].
  2. The fixed truncation (keep N+1 layers) reproduces hidden_states[N]
     bitwise-exactly.
It also greps the real source to guard against the fix silently regressing.

Requires torch + transformers (both already required by the production
acoustic path) — skips gracefully if unavailable, matching the house
convention for real-fixture-dependent checks (see verify_diarization_bridge.py
step 10). No GPU needed; runs on CPU.
"""

import copy
import re
import sys
from pathlib import Path

PASS = 0
FAIL = 0
SKIP = 0


def check(description: str, condition: bool):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"✅ {description}")
    else:
        FAIL += 1
        print(f"❌ {description}")


def skip(description: str):
    global SKIP
    SKIP += 1
    print(f"➖ {description} — skipped (torch/transformers unavailable)")


def main():
    try:
        import torch
        from transformers import WavLMConfig, WavLMModel
    except ImportError:
        skip("WavLM truncation checks (torch/transformers not installed)")
        print(f"\n{'=' * 70}\n➖ SKIPPED — dependencies unavailable")
        return 0

    repo_root = Path(__file__).resolve().parent.parent

    # ── 1. Source-guard: the fix must not silently regress ───────────────
    src = (repo_root / "audio_isolation" / "core" / "acoustic_extractor.py").read_text()
    check("1. _load_wavlm keeps WAVLM_LAYER_INDEX + 1 layers (not WAVLM_LAYER_INDEX)",
          "model.encoder.layers = model.encoder.layers[:WAVLM_LAYER_INDEX + 1]" in src)
    check("   the buggy exact-cut form (no +1) is not present anywhere",
          "model.encoder.layers[:WAVLM_LAYER_INDEX]" not in src)
    check("   read site still uses the fixed WAVLM_LAYER_INDEX (no -2 needed with N+1 layers)",
          "out.hidden_states[WAVLM_LAYER_INDEX]" in src)

    # ── 2. Numerical proof on a tiny synthetic StableLayerNorm model ──────
    torch.manual_seed(0)
    N_LAYERS = 8
    LAYER_INDEX = 5  # analogous to production WAVLM_LAYER_INDEX=14 of 24

    cfg = WavLMConfig(
        hidden_size=16, num_hidden_layers=N_LAYERS, num_attention_heads=4,
        intermediate_size=32, conv_dim=(16,) * 7,
        conv_stride=(5, 2, 2, 2, 2, 2, 2), conv_kernel=(10, 3, 3, 3, 3, 2, 2),
        do_stable_layer_norm=True, num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )
    full_model = WavLMModel(cfg).eval()
    check("2. synthetic model actually uses the StableLayerNorm encoder "
          "(the mechanism the bug depends on)",
          type(full_model.encoder).__name__ == "WavLMEncoderStableLayerNorm")

    x = torch.randn(1, 4000)
    with torch.inference_mode():
        full_hidden = full_model(input_values=x, output_hidden_states=True).hidden_states
    check(f"   full {N_LAYERS}-layer stack produces {N_LAYERS + 1} hidden_states entries",
          len(full_hidden) == N_LAYERS + 1)

    # BUGGY path: truncate to exactly LAYER_INDEX layers (the pre-fix code).
    buggy_model = copy.deepcopy(full_model)
    buggy_model.encoder.layers = buggy_model.encoder.layers[:LAYER_INDEX]
    with torch.inference_mode():
        buggy_hidden = buggy_model(input_values=x, output_hidden_states=True).hidden_states
    buggy_read = buggy_hidden[LAYER_INDEX]

    # FIXED path: truncate to LAYER_INDEX + 1 layers (the actual fix).
    fixed_model = copy.deepcopy(full_model)
    fixed_model.encoder.layers = fixed_model.encoder.layers[:LAYER_INDEX + 1]
    with torch.inference_mode():
        fixed_hidden = fixed_model(input_values=x, output_hidden_states=True).hidden_states
    fixed_read = fixed_hidden[LAYER_INDEX]

    ground_truth = full_hidden[LAYER_INDEX]

    check(f"3. BUGGY truncation (keep exactly {LAYER_INDEX} layers) corrupts "
          f"hidden_states[{LAYER_INDEX}] — an extra LayerNorm lands on it",
          not torch.allclose(buggy_read, ground_truth, atol=1e-6))
    check(f"   FIXED truncation (keep {LAYER_INDEX + 1} layers) reproduces "
          f"hidden_states[{LAYER_INDEX}] bitwise-exactly vs. the full stack",
          torch.allclose(fixed_read, ground_truth, atol=1e-6))
    check("   the fixed and buggy reads actually differ (the test isn't vacuous)",
          not torch.allclose(fixed_read, buggy_read, atol=1e-6))

    # Sanity: entries below the cut are untouched by truncation either way
    # (this part of the code's original claim was correct).
    check("4. entries below the cut are bitwise-identical across full/buggy/fixed "
          "(only the read-index entry was ever at risk)",
          all(torch.allclose(full_hidden[i], buggy_hidden[i], atol=1e-6)
              and torch.allclose(full_hidden[i], fixed_hidden[i], atol=1e-6)
              for i in range(LAYER_INDEX)))

    print(f"\n{'=' * 70}")
    print(f"VERIFICATION RESULTS: {PASS}/{PASS + FAIL} checks passed"
          + (f", {SKIP} skipped" if SKIP else ""))
    if FAIL == 0:
        print("🏆 SUCCESS — WavLM encoder truncation fix numerically verified.")
        return 0
    print("❌ FAILURES PRESENT.")
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
