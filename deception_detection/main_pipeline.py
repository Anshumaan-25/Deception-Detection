import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path

# --- Ingestion & Synchronization Hooks ---
from opencv_streaming.core.stream_reader import CanonicalStreamReader

# --- Feature Extraction Engines ---
from mediapipe_pose.parallel_pool import ParallelMediaPipePool
from openface_pipeline.api.extractor import OpenFaceExtractor

# --- Audio Isolation Engine ---
from audio_isolation.core.diarizer_engine import MediaPipeAudioDiarizer
from audio_isolation.core.diarization_bridge import DiarizationBridge

# --- Acoustic Feature Extraction (WavLM) ---
from audio_isolation.core.acoustic_extractor import WavLMAcousticExtractor, WAVLM_MODEL_NAME, WAVLM_LAYER_INDEX

# --- Downstream Analytics Engines ---
from analytics.dynamic_window_engine import DynamicWindowEngine
from analytics.baseline_calibrator import BaselineCalibrator, BaselineCalibrationError
from analytics.recording_assembler import assemble_recording
from analytics.context_mapper import ContextMapper

# --- Headless Tracking Hooks ---
from Yolo_v8.PersonTracking4.src.detector import PersonDetector
from Yolo_v8.PersonTracking4.src.face_lock import FaceLock


class MultimodalProductionOrchestrator:
    def __init__(self, output_root: str, yolo_path: str, attenuation: float = 0.05):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        print("🚀 Booting Production Workstation Pipelines (TensorRT FP16 Optimized)...")
        
        # ── TensorRT Engine Resolution ───────────────────────────────
        # Resolve the compiled .engine path from the original .pt path.
        # If a .engine file exists alongside the .pt, route through TensorRT.
        # Otherwise, fall back to the .pt path (PyTorch/CUDA runtime).
        yolo_engine_path = yolo_path.replace('.pt', '.engine')
        if os.path.exists(yolo_engine_path):
            print(f"   🚀 YOLOv8: TensorRT engine detected → {yolo_engine_path}")
            active_yolo_path = yolo_engine_path
        else:
            print(f"   ⚠️  YOLOv8: No .engine found, using PyTorch → {yolo_path}")
            active_yolo_path = yolo_path
        
        # Resolve InsightFace TRT engine cache directory
        self.trt_cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "weights", "trt_engines", "insightface"
        )
        
        self.parallel_pool = ParallelMediaPipePool(num_workers=12)
        self.openface_pipeline = OpenFaceExtractor(legacy_weights_dir="openface_pipeline/weights")
        self.audio_pipeline = MediaPipeAudioDiarizer(attenuation_factor=attenuation)
        
        self.detector = PersonDetector(model_path=active_yolo_path, device='cuda')
        self.facelock = FaceLock(engine_cache_dir=self.trt_cache_dir)
        self.target_locked = False

    def compile_raw_features(self, pose_data: list, openface_data: list, output_csv_path: str):
        """
        Vectorizes kinematic math in-memory and outputs a pure 30 FPS frame-by-frame CSV.
        Leaves temporal windowing entirely to the downstream Layer 4 engine.
        """
        if not pose_data or not openface_data:
            print("⚠️ WARNING: Insufficient data for fusion.")
            return None

        df_pose = pd.DataFrame(pose_data)
        df_of = pd.DataFrame(openface_data)

        # --- 1. Vectorized Kinematic Feature Engineering (In-Memory) ---
        # Hand-to-Face Distance (Full 3D Euclidean — includes depth component)
        df_pose["left_hand_face_distance"] = np.sqrt(
            (df_pose["left_wrist_x"] - df_pose["nose_x"])**2 +
            (df_pose["left_wrist_y"] - df_pose["nose_y"])**2 +
            (df_pose["left_wrist_z"] - df_pose["nose_z"])**2
        )
        df_pose["right_hand_face_distance"] = np.sqrt(
            (df_pose["right_wrist_x"] - df_pose["nose_x"])**2 +
            (df_pose["right_wrist_y"] - df_pose["nose_y"])**2 +
            (df_pose["right_wrist_z"] - df_pose["nose_z"])**2
        )
        
        # Wrist Velocity (First Derivative — 3D Euclidean, NaN on tracking gaps)
        df_pose["left_wrist_velocity"] = np.sqrt(
            df_pose["left_wrist_x"].diff()**2 +
            df_pose["left_wrist_y"].diff()**2 +
            df_pose["left_wrist_z"].diff()**2
        )
        df_pose["right_wrist_velocity"] = np.sqrt(
            df_pose["right_wrist_x"].diff()**2 +
            df_pose["right_wrist_y"].diff()**2 +
            df_pose["right_wrist_z"].diff()**2
        )
        
        # Motion Energy — uses the correct 7-landmark macro_motion_energy
        # from the parallel pool, NOT a crude 2-wrist proxy sum.
        # macro_motion_energy is already present in pose_data from parallel_pool.py.

        # --- 2. Frame-by-Frame Inner Join (No Aggregation) ---
        # Because we use the Master Clock, both dataframes share exact timestamps
        # We perform an exact row-by-row merge at 30 FPS
        fused_frames = pd.merge(df_pose, df_of, on="timestamp", how="inner")

        # --- Vectorized Gaze Velocity ---
        # First-derivative tracking of 2D gaze vectors. Drops back to np.nan on missing tracking.
        fused_frames["gaze_velocity"] = np.sqrt(fused_frames["gaze_x"].diff()**2 + fused_frames["gaze_y"].diff()**2)

        # --- 3. AU Onset Velocity (First Derivative) ---
        # Captures micro-expression onset speed at 30fps.
        # Genuine expressions: ~250-500ms onset. Posed: too fast or too slow.
        au_columns = ["AU1", "AU2", "AU4", "AU6", "AU9", "AU12", "AU25", "AU26"]
        for au in au_columns:
            fused_frames[f"{au}_velocity"] = fused_frames[au].diff().fillna(0)

        # --- 4. Vectorized Joint Confidence Vector (w_t) Computation ---
        # w_t = c_yolo * c_facelock * c_landmark * c_diarizer
        # Replace any NaN confidences with 0.0 before computing product to ensure strict safety
        yolo_c = fused_frames["yolo_conf"].fillna(0.0)
        fl_c = fused_frames["facelock_conf"].fillna(0.0)
        face_c = fused_frames["face_confidence"].fillna(0.0)
        diar_c = fused_frames["diarizer_conf"].fillna(0.0)
        
        fused_frames["joint_confidence"] = yolo_c * fl_c * face_c * diar_c

        # --- 5. Save Raw High-Resolution Tensor ---
        fused_frames.to_csv(output_csv_path, index=False)
        print(f"✅ Raw 30 FPS Tensor compiled: {len(fused_frames)} synchronized frames generated.")
        return output_csv_path

    def _run_yolo(self, chunk: list):
        if not chunk:
            return []
        frames = [item[2] for item in chunk]
        return self.detector.detect_and_track_batch(frames)

    def _process_chunk_tail(self, chunk: list, batch_detections: list, openface_records: list):
        """
        Executes CPU-bound FaceLock and GPU-bound MLT extraction for a tracked chunk.
        """
        if not chunk:
            return

        frames = [item[2] for item in chunk]
        frame_ids = [item[0] for item in chunk]
        timestamps = [item[1] for item in chunk]

        # 2. Sequential CPU FaceLock Matching & Cropping
        crops_list = []
        chunk_confidences = [] # list of (yolo_conf, facelock_conf) for each frame in the chunk

        for idx, frame in enumerate(frames):
            fid = frame_ids[idx]
            ts = timestamps[idx]
            detections = batch_detections[idx]

            if not detections:
                self.parallel_pool.mark_gap(fid, ts)
                crops_list.append(None)
                chunk_confidences.append((0.0, 0.0))
                continue

            # Auto-Locking
            yolo_conf = 0.0
            facelock_conf = 0.0
            if not self.target_locked:
                locked_bbox = None
                for d in detections:
                    if self.facelock.lock(frame, d["bbox"]):
                        print(f"🎯 Auto-Lock Engaged: Target ID {d['id']} at {ts}ms")
                        self.target_locked = True
                        locked_bbox = d["bbox"]
                        yolo_conf = d.get("conf", 1.0)
                        facelock_conf = 1.0  # Anchor is perfect lock match
                        break
            else:
                locked_bbox, facelock_conf = self.facelock.match(frame, detections)
                if locked_bbox is not None:
                    # Find yolo confidence for the locked bbox
                    for d in detections:
                        if d["bbox"] == locked_bbox:
                            yolo_conf = d.get("conf", 1.0)
                            break

            if locked_bbox is not None:
                x1, y1, x2, y2 = locked_bbox
                h, w, _ = frame.shape
                crop_frame = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                if crop_frame.size == 0:
                    self.parallel_pool.mark_gap(fid, ts)
                    crops_list.append(None)
                    chunk_confidences.append((0.0, 0.0))
                    continue

                # Convert BGR (OpenCV default) → RGB for MediaPipe accuracy
                rgb_crop = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
                
                # Submit to Parallel MediaPipe Pool (async CPU workers)
                self.parallel_pool.submit_task(fid, ts, rgb_crop)
                crops_list.append(crop_frame)
                chunk_confidences.append((yolo_conf, facelock_conf))
            else:
                self.parallel_pool.mark_gap(fid, ts)
                crops_list.append(None)
                chunk_confidences.append((0.0, 0.0))

        # 3. Batched OpenFace Extraction (MLT)
        of_batch_results = self.openface_pipeline.process_batch(crops_list, frame_ids, timestamps)

        # Defensive Synchronization Barrier: 
        # Guarantee MLT CUDA stream is aligned before appending records.
        import torch
        self.openface_pipeline.mlt_stream.synchronize()

        # 4. Parse and append to openface_records
        for idx, of_result in enumerate(of_batch_results):
            ts = timestamps[idx]
            yolo_conf, facelock_conf = chunk_confidences[idx]
            
            if of_result and of_result["faces"]:
                face_data = of_result["faces"][0]
                openface_records.append({
                    "timestamp": ts,
                    "emotion_label": face_data["emotion"]["primary_label"],
                    "emotion_confidence": face_data["emotion"]["confidence"],
                    "face_confidence": face_data.get("confidence", 1.0),
                    "yolo_conf": yolo_conf,
                    "facelock_conf": facelock_conf,
                    "gaze_x": face_data["gaze_3d"][0],
                    "gaze_y": face_data["gaze_3d"][1],
                    "gaze_z": face_data["gaze_3d"][2],
                    "AU1": face_data["action_units"]["AU1"],
                    "AU2": face_data["action_units"]["AU2"],
                    "AU4": face_data["action_units"]["AU4"],
                    "AU6": face_data["action_units"]["AU6"],
                    "AU9": face_data["action_units"]["AU9"],
                    "AU12": face_data["action_units"]["AU12"],
                    "AU25": face_data["action_units"]["AU25"],
                    "AU26": face_data["action_units"]["AU26"],
                })
            else:
                openface_records.append({
                    "timestamp": ts,
                    "emotion_label": "N/A",
                    "emotion_confidence": 0.0,
                    "face_confidence": 0.0,
                    "yolo_conf": 0.0,
                    "facelock_conf": 0.0,
                    "gaze_x": np.nan, "gaze_y": np.nan, "gaze_z": np.nan,
                    "AU1": np.nan, "AU2": np.nan, "AU4": np.nan,
                    "AU6": np.nan, "AU9": np.nan, "AU12": np.nan,
                    "AU25": np.nan, "AU26": np.nan,
                })

    def process_video_session(self, canonical_mp4_path: str, canonical_wav_path: str, session_id: str, session_manifest_path: str = None, pyannote_segments: list = None, calibrate: bool = True):
        """
        Unified End-to-End Session Orchestrator.

        Executes the complete multimodal data cascade for a single subject
        session in 4 sequential phases:

            Phase 1 — Visual & Audio Extraction:
                Stream video frames through B=16 GPU chunks (YOLOv8 TensorRT
                tracking → FaceLock identity → 12-worker MediaPipe pool +
                OpenFace MLT). Execute cross-modal speaker isolation and
                boot WavLM acoustic microservice on isolated audio.

            Phase 2 — Raw Feature Compilation:
                Fuse pose, facial, and gaze records into a synchronized 30fps
                frame-by-frame tensor with joint_confidence vector.

            Phase 3 — Sliding Window Aggregation:
                DynamicWindowEngine (2s windows, 1s stride) applies confidence-
                weighted regularization, injects WavLM acoustic features,
                and computes 36-column FFT behavioral periodicity metrics.

            Phase 4 — Baseline Calibration:
                Z-score normalizes all windowed features against the subject's
                first 30-second neutral baseline period, producing deviation
                magnitude and percentile rank columns.

        Each phase is individually error-bounded. Upstream outputs are
        preserved even if a downstream stage fails.
        """
        # Reset FaceLock for each new session to prevent embedding leakage across subjects
        self.facelock = FaceLock(engine_cache_dir=self.trt_cache_dir)
        self.target_locked = False

        session_dir = self.output_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = session_dir / "metadata.json"

        # ── Manifest Ledger ──────────────────────────────────────────
        master_manifest = {
            "session_id": session_id,
            "status": "processing",
            "stages": {},
            "outputs": {},
        }

        openface_records = []
        batch_size = 16
        batch_buffer = []

        # Output path contracts (absolute paths for manifest)
        fused_csv_path = session_dir / f"{session_id}_raw_features_30fps.csv"
        isolated_wav_path = session_dir / "audio_isolation" / "isolated_target_audio.wav"
        windowed_csv_path = session_dir / f"{session_id}_windowed_features.csv"
        calibrated_csv_path = session_dir / f"{session_id}_calibrated_features.csv"

        print(f"\n{'='*60}")
        print(f"  UNIFIED PIPELINE — SESSION: {session_id}")
        print(f"{'='*60}")

        from concurrent.futures import ThreadPoolExecutor

        executor = ThreadPoolExecutor(max_workers=4)
        future_yolo = None
        prev_chunk = None

        try:
            # ═════════════════════════════════════════════════════════
            # PHASE 1: Visual & Audio Extraction
            # ═════════════════════════════════════════════════════════
            print(f"\n--- PHASE 1: HIGH-SPEED RAM INGESTION ---")

            with CanonicalStreamReader(canonical_mp4_path) as streamer:
                for frame_id, timestamp_ms, frame in streamer.stream_frames():
                    batch_buffer.append((frame_id, timestamp_ms, frame))
                    if len(batch_buffer) == batch_size:
                        current_chunk = batch_buffer
                        batch_buffer = []

                        # Submit Chunk N+1 to YOLO tracking Thread
                        next_future_yolo = executor.submit(self._run_yolo, current_chunk)

                        if prev_chunk is not None:
                            # Wait for YOLO on Chunk N
                            batch_detections = future_yolo.result()
                            # Process FaceLock & MLT on Chunk N (Double-Buffered)
                            self._process_chunk_tail(prev_chunk, batch_detections, openface_records)
                        
                        prev_chunk = current_chunk
                        future_yolo = next_future_yolo

                # Handle trailing frames gracefully
                if batch_buffer:
                    next_future_yolo = executor.submit(self._run_yolo, batch_buffer)
                    if prev_chunk is not None:
                        batch_detections = future_yolo.result()
                        self._process_chunk_tail(prev_chunk, batch_detections, openface_records)
                    prev_chunk = batch_buffer
                    future_yolo = next_future_yolo

                # Flush the last remaining chunk
                if prev_chunk is not None:
                    batch_detections = future_yolo.result()
                    self._process_chunk_tail(prev_chunk, batch_detections, openface_records)

            print("\n--- INGESTION COMPLETE. COLLECTING PARALLEL RESULTS ---")

            # O(1) Streaming Re-ordering Buffer: collects all worker results,
            # sorts chronologically, computes temporal derivatives (velocities, energies)
            pose_records, visual_lip_logs = self.parallel_pool.collect_results()

            print(f"✅ Parallel pool returned {len(pose_records)} pose records, {len(visual_lip_logs)} lip logs.")

            master_manifest["stages"]["visual_extraction"] = {
                "status": "success",
                "pose_records": len(pose_records),
                "lip_logs": len(visual_lip_logs),
            }

            # ── Audio Isolation & Cross-Modal Identity Anchoring ─────
            print("--- EXECUTING AUDIO ISOLATION ---")

            audio_isolation_dir = session_dir / "audio_isolation"

            # ── MERGE SEAM (audio diarization → deception detection) ──
            # Real verified-target segments from the audio-diarization pipeline,
            # adapted by DiarizationBridge to [("TARGET", start_ms, end_ms), ...].
            # Fallback semantics:
            #   pyannote_segments is None → no diarization wired → legacy
            #       whole-clip mock, so existing single-clip runs are unchanged.
            #   pyannote_segments == []   → diarization ran but the target is
            #       silent in this clip → anchor returns None → TARGET_SILENT.
            if pyannote_segments is None:
                pyannote_segments = [("SPEAKER_00", 0.0, timestamp_ms)]

            # Hybrid Cross-Modal Identity Anchoring & Incongruence post-processing
            target_id, correlation_score = self.audio_pipeline.anchor_target_identity(
                pyannote_segments, visual_lip_logs
            )
            if target_id is None:
                target_id = "TARGET_SILENT"
                correlation_score = 0.0

            target_segments = [seg for seg in pyannote_segments if seg[0] == target_id]

            for pose_rec, lip_rec in zip(pose_records, visual_lip_logs):
                t_ms = pose_rec["timestamp"]
                is_audio_active = 0.0
                for _, start_ms, end_ms in target_segments:
                    if start_ms <= t_ms < end_ms:
                        is_audio_active = 1.0
                        break

                is_lips_active = float(lip_rec.get("is_moving", 0))

                if np.isnan(pose_rec.get("nose_x", np.nan)):
                    mismatch = np.nan
                    silent_speech = np.nan
                    diarizer_conf = np.nan
                else:
                    mismatch = 1.0 if (is_audio_active == 1.0 and is_lips_active == 0.0) else 0.0
                    silent_speech = 1.0 if (is_lips_active == 1.0 and is_audio_active == 0.0) else 0.0
                    # Diarizer confidence: 0.90 if speaking, 1.0 if silent.
                    diarizer_conf = 0.9 if is_audio_active == 1.0 else 1.0

                pose_rec["is_audio_active"] = is_audio_active
                pose_rec["mismatch_incongruence"] = mismatch
                pose_rec["silent_incongruence"] = silent_speech
                pose_rec["diarizer_conf"] = diarizer_conf

            self.audio_pipeline.execute_isolation_pipeline(
                input_wav_path=canonical_wav_path,
                visual_speech_logs=visual_lip_logs,
                pyannote_segments=pyannote_segments,
                output_dir=str(audio_isolation_dir)
            )

            master_manifest["stages"]["audio_isolation"] = {
                "status": "success",
                "target_speaker": target_id,
                "correlation_score": float(correlation_score),
            }
            master_manifest["outputs"]["isolated_target_audio"] = str(isolated_wav_path)

            # ── Boot WavLM Acoustic Feature Extractor ───────────────
            print("--- BOOTING WAVLM ACOUSTIC MICROSERVICE ---")
            acoustic_extractor = WavLMAcousticExtractor(str(isolated_wav_path))

            master_manifest["stages"]["acoustic_extraction"] = {
                "status": "success",
                "model": WAVLM_MODEL_NAME,
                "layer": WAVLM_LAYER_INDEX,
                "audio_duration_ms": acoustic_extractor.total_duration_ms,
            }

            # ═════════════════════════════════════════════════════════
            # PHASE 2: Raw Feature Compilation
            # ═════════════════════════════════════════════════════════
            print(f"\n--- PHASE 2: RAW FEATURE COMPILATION ---")

            raw_result = self.compile_raw_features(
                pose_records, openface_records, str(fused_csv_path)
            )

            if raw_result is None:
                raise RuntimeError(
                    "Raw feature compilation failed — insufficient data for fusion. "
                    "Check pose and OpenFace record counts."
                )

            # Read back to count fused frames for manifest
            fused_frame_count = len(pd.read_csv(fused_csv_path))

            master_manifest["stages"]["raw_compilation"] = {
                "status": "success",
                "total_fused_frames": fused_frame_count,
            }
            master_manifest["outputs"]["raw_features_30fps"] = str(fused_csv_path)

            # ═════════════════════════════════════════════════════════
            # PHASE 3: Sliding Window Aggregation
            #   → Confidence-Weighted Fusion (Target #15)
            #   → WavLM Acoustic Injection (Target #13)
            #   → FFT Behavioral Periodicity (Target #16)
            # ═════════════════════════════════════════════════════════
            print(f"\n--- PHASE 3: WINDOW AGGREGATION (Confidence + Acoustic + FFT) ---")

            window_engine = DynamicWindowEngine(
                window_size_ms=2000.0,
                stride_ms=1000.0,
                min_fill_rate=0.25,
                assumed_fps=30.0,
                min_confidence_threshold=0.35,
            )

            context_mapper = ContextMapper(manifest_path=session_manifest_path)

            windowed_result = window_engine.compile_sliding_windows(
                raw_csv_path=str(fused_csv_path),
                output_csv_path=str(windowed_csv_path),
                acoustic_extractor=acoustic_extractor,
                context_mapper=context_mapper,
            )

            if windowed_result is None:
                raise RuntimeError(
                    "Window aggregation failed — raw CSV may be missing or empty."
                )

            # Read back to count windows for manifest
            windowed_df = pd.read_csv(windowed_csv_path)
            total_windows = len(windowed_df)
            total_feature_cols = len([
                c for c in windowed_df.columns
                if c not in ("window_id", "start_time_ms", "end_time_ms",
                             "frame_count", "cumulative_confidence",
                             "emotion_label_mode")
            ])

            master_manifest["stages"]["window_aggregation"] = {
                "status": "success",
                "total_windows": total_windows,
                "feature_columns": total_feature_cols,
            }
            master_manifest["outputs"]["windowed_features"] = str(windowed_csv_path)

            print(f"✅ Window Aggregation Complete: {total_windows} windows, {total_feature_cols} feature columns.")

            # ═════════════════════════════════════════════════════════
            # PHASE 4: Baseline Calibration (Z-Score Normalization)
            #   Skipped in recording mode: the orchestrator calibrates
            #   every clip against the dedicated baseline clip's stats
            #   (see process_recording_session), not each clip against
            #   its own first 30 seconds.
            # ═════════════════════════════════════════════════════════
            if calibrate:
                print(f"\n--- PHASE 4: BASELINE CALIBRATION ---")

                calibrator = BaselineCalibrator(calibration_duration_ms=30000.0)

                calibrated_result = calibrator.calibrate(
                    windowed_csv_path=str(windowed_csv_path),
                    output_csv_path=str(calibrated_csv_path),
                )

                if calibrated_result is None:
                    raise RuntimeError(
                        "Baseline calibration failed — windowed CSV may be missing or empty."
                    )

                # Read back to count calibrated features for manifest
                calibrated_df = pd.read_csv(calibrated_csv_path)
                baseline_mask = calibrated_df["start_time_ms"] < 30000.0
                baseline_window_count = int(baseline_mask.sum())

                master_manifest["stages"]["baseline_calibration"] = {
                    "status": "success",
                    "baseline_windows": baseline_window_count,
                    "calibrated_features": len(calibrated_df.columns),
                }
                master_manifest["outputs"]["calibrated_features"] = str(calibrated_csv_path)

                print(f"✅ Baseline Calibration Complete: {baseline_window_count} baseline windows used.")
                final_output_path = calibrated_csv_path
            else:
                print(f"\n--- PHASE 4: SKIPPED (recording mode — calibration deferred to orchestrator) ---")
                baseline_window_count = "deferred"
                master_manifest["stages"]["baseline_calibration"] = {
                    "status": "skipped",
                    "reason": "recording_mode_deferred_to_orchestrator",
                }
                final_output_path = windowed_csv_path

            # ═════════════════════════════════════════════════════════
            # PIPELINE COMPLETE
            # ═════════════════════════════════════════════════════════
            master_manifest["status"] = "success"

            print(f"\n{'='*60}")
            print(f"  🏆 UNIFIED PIPELINE COMPLETE — {session_id}")
            print(f"     Raw Frames:       {fused_frame_count}")
            print(f"     Windows:          {total_windows}")
            print(f"     Feature Columns:  {total_feature_cols}")
            print(f"     Baseline Windows: {baseline_window_count}")
            print(f"     Output:           {final_output_path}")
            print(f"{'='*60}")

        except Exception as e:
            master_manifest["status"] = "failed"
            master_manifest["error_log"] = str(e)
            print(f"\n❌ Pipeline Failure at stage boundary: {str(e)}")
            import traceback
            traceback.print_exc()

        finally:
            if 'executor' in locals():
                executor.shutdown(wait=True)
            # Persist manifest — always executes, even on failure
            with open(manifest_path, "w") as f:
                json.dump(master_manifest, f, indent=4)
            self.parallel_pool.close()
            print(f"📋 Manifest written to: {manifest_path}")

    def process_recording_session(self, clips, recording_id, diarization_output_json,
                                  *, offset_ms: int = 0, clock: str = "local",
                                  session_manifest_path: str = None,
                                  baseline_file_index: int = 0):
        """
        Batch entrypoint (audio-diarization merge + Phase A calibration).

        One recording = N clips that share a single audio-diarization
        enrollment, where the clip at ``baseline_file_index`` (default 0) is
        the DEDICATED BASELINE VIDEO: the target answering generic/neutral
        questions before the interview proper. The flow:

            1. Per-clip cascade (extraction + windowing) for every clip, with
               per-clip self-calibration SKIPPED (calibrate=False).
            2. Fit baseline stats on the baseline clip's windowed CSV — every
               window of the whole clip, no duration cap. Fails loudly
               (BaselineCalibrationError) if the baseline clip is unusable:
               a recording without a baseline has no meaningful deviations.
            3. Apply those stats to every clip (baseline included — its own
               deviations land near 0, a built-in sanity check).
            4. Assemble the recording-level CSV: rebase per-clip times by the
               diarization file_offset_ms, concatenate, renumber window_ids,
               rank deviation_percentile over the whole recording.

        Windows never cross clip boundaries: windowing runs per clip, so the
        hard break is automatic.

        Args:
            clips: list of (canonical_mp4_path, canonical_wav_path) pairs, in
                any order; each is mapped to its diarization file_index by stem.
            recording_id: id for this recording. Per-clip session ids are
                f"{recording_id}_{file_index:03d}".
            diarization_output_json: path to the audio-diarization
                pipeline_output.json produced for this recording's clip batch.
            offset_ms: per-recording time-alignment offset (MERGE_INTEGRATION_PLAN
                §7); default 0.
            clock: "local" — per-clip timeline (correct for per-clip processing).
            session_manifest_path: optional investigative session manifest for
                ContextMapper phase/question labels (previously never wired in
                batch mode — pre-existing bug, fixed here).
            baseline_file_index: which diarization file_index is the baseline
                video. Default 0 (baseline clip named to sort first in the
                diarization batch's canonical order); override via
                session_profile.json for mis-named batches.

        Returns a dict: per-clip results plus the recording-level artifact paths.
        """
        bridge = DiarizationBridge.from_output_json(diarization_output_json)

        # ── Pass 1: per-clip cascade, calibration deferred ────────────
        processed = []
        for canonical_mp4, canonical_wav in clips:
            file_index = bridge.index_for_clip(canonical_mp4)
            segments = bridge.segments_for(file_index, clock=clock, offset_ms=offset_ms)
            session_id = f"{recording_id}_{file_index:03d}"
            role = "BASELINE" if file_index == baseline_file_index else "interview"
            print(f"\n>> Recording '{recording_id}': clip file_index={file_index} ({role}), "
                  f"{len(segments)} target segment(s) → session {session_id}")
            self.process_video_session(
                canonical_mp4_path=str(canonical_mp4),
                canonical_wav_path=str(canonical_wav),
                session_id=session_id,
                session_manifest_path=session_manifest_path,
                pyannote_segments=segments,
                calibrate=False,
            )
            session_dir = self.output_root / session_id
            processed.append({
                "file_index": file_index,
                "session_id": session_id,
                "segment_count": len(segments),
                "windowed_csv": session_dir / f"{session_id}_windowed_features.csv",
            })
        processed.sort(key=lambda r: r["file_index"])

        # ── Pass 2: fit baseline on the dedicated calibration clip ───
        baseline = next(
            (r for r in processed if r["file_index"] == baseline_file_index), None
        )
        if baseline is None:
            raise BaselineCalibrationError(
                f"Recording '{recording_id}': no clip has baseline_file_index="
                f"{baseline_file_index}; indices present: "
                f"{[r['file_index'] for r in processed]}"
            )
        calibrator = BaselineCalibrator()
        stats = calibrator.fit(str(baseline["windowed_csv"]))  # raises if unusable

        recording_dir = self.output_root / recording_id
        recording_dir.mkdir(parents=True, exist_ok=True)
        stats_path = recording_dir / f"{recording_id}_baseline_stats.json"
        stats.to_json(str(stats_path))

        # ── Pass 3: apply baseline stats to every clip ────────────────
        for rec in processed:
            if not rec["windowed_csv"].exists():
                # Non-baseline clip failed upstream: skip it in assembly but
                # keep the recording alive (its manifest records the failure).
                print(f"⚠️ Recording '{recording_id}': windowed CSV missing for "
                      f"{rec['session_id']} — clip excluded from assembly.")
                continue
            calibrated_csv = (self.output_root / rec["session_id"]
                              / f"{rec['session_id']}_calibrated_features.csv")
            calibrator.apply(str(rec["windowed_csv"]), stats, str(calibrated_csv))
            rec["calibrated_csv"] = calibrated_csv

        # ── Pass 4: assemble the recording-level CSV ──────────────────
        assembly_inputs = [
            {
                "file_index": rec["file_index"],
                "csv_path": str(rec["calibrated_csv"]),
                "offset_ms": bridge.file_offset_ms(rec["file_index"]),
            }
            for rec in processed if "calibrated_csv" in rec
        ]
        recording_csv = recording_dir / f"{recording_id}_recording_calibrated.csv"
        assemble_recording(assembly_inputs, str(recording_csv))

        print(f"\n🏁 Recording '{recording_id}' complete: "
              f"{len(assembly_inputs)}/{len(processed)} clip(s) assembled, "
              f"baseline={stats.baseline_window_count} windows (file_index "
              f"{baseline_file_index}).")
        return {
            "recording_id": recording_id,
            "baseline_file_index": baseline_file_index,
            "baseline_stats_json": str(stats_path),
            "recording_calibrated_csv": str(recording_csv),
            "clips": [
                {k: str(v) if isinstance(v, Path) else v for k, v in rec.items()}
                for rec in processed
            ],
        }


if __name__ == "__main__":
    orchestrator = MultimodalProductionOrchestrator(
        output_root="pipeline_system_outputs",
        yolo_path="weights/yolov8n.pt"
    )
    # UNCOMMENT TO RUN YOUR ASSET:
    # orchestrator.process_video_session("path/to/canonical.mp4", "path/to/canonical.wav", "SESSION_001")