import os
import json
import numpy as np
from scipy.io import wavfile

class MediaPipeAudioDiarizer:
    def __init__(self, attenuation_factor=0.05):
        """
        Multimodal Audio Diarizer & Target Voice Isolator.
        Fuses acoustic speaker clustering timestamps with continuous 
        visual lip tracking matrices to isolate a specific target subject.
        """
        self.attenuation_factor = attenuation_factor
        
    def anchor_target_identity(self, pyannote_segments, visual_speech_logs):
        """
        Executes Hybrid Cross-Modal Identity Anchoring via NumPy Vectorization.
        """
        if not pyannote_segments or not visual_speech_logs:
            return None, 0.0

        # --- FIX 2: Vectorize the Python dictionary list into high-speed NumPy arrays ---
        timestamps = np.array([f['timestamp_ms'] for f in visual_speech_logs])
        is_moving = np.array([f['is_moving'] for f in visual_speech_logs])
        velocities = np.array([f['mor_velocity'] for f in visual_speech_logs])

        speaker_metrics = {}
        
        for speaker_id, start_ms, end_ms in pyannote_segments:
            if speaker_id not in speaker_metrics:
                speaker_metrics[speaker_id] = {
                    "binary_matches": 0,
                    "cumulative_velocity": 0.0,
                    "total_frames_in_window": 0
                }
            
            # --- FIX 1 & 2: Half-open interval (<) and vectorized C-backend masking ---
            window_mask = (timestamps >= start_ms) & (timestamps < end_ms)
            
            # Sum up values instantly without loops
            speaker_metrics[speaker_id]["binary_matches"] += np.sum(is_moving[window_mask])
            speaker_metrics[speaker_id]["cumulative_velocity"] += np.sum(velocities[window_mask])
            speaker_metrics[speaker_id]["total_frames_in_window"] += np.sum(window_mask)

        target_speaker_id = None
        highest_composite_score = -1.0
        
        # Determine target identity via normalized hybrid scoring optimization
        for speaker_id, data in speaker_metrics.items():
            if data["total_frames_in_window"] == 0:
                continue
                
            binary_overlap_ratio = float(data["binary_matches"] / data["total_frames_in_window"])
            mean_visual_velocity = float(data["cumulative_velocity"] / data["total_frames_in_window"])
            
            composite_score = binary_overlap_ratio * mean_visual_velocity
            
            if composite_score > highest_composite_score:
                highest_composite_score = composite_score
                target_speaker_id = speaker_id
                
        return target_speaker_id, highest_composite_score

    def isolate_voice_channel(self, input_wav_path, output_wav_path, pyannote_segments, target_speaker_id):
        """
        Vectorized Mask Attenuation.
        Multiplies all non-target acoustic intervals by a scalar factor to drop
        the interviewer's track into the noise floor.
        """
        sample_rate, audio_signal = wavfile.read(input_wav_path)
        
        # Guard: Ensure canonical mono enforcement (1D array)
        if audio_signal.ndim > 1:
            audio_signal = audio_signal[:, 0]
            
        target_audio_mask = np.zeros(audio_signal.shape[0], dtype=bool)
        
        # If target_speaker_id is None, mask remains False, attenuating the whole file
        if target_speaker_id is not None:
            for speaker_id, start_ms, end_ms in pyannote_segments:
                if speaker_id == target_speaker_id:
                    start_index = int((start_ms / 1000.0) * sample_rate)
                    end_index = int((end_ms / 1000.0) * sample_rate)
                    
                    # Protect matrix bounds
                    start_index = max(0, start_index)
                    end_index = min(audio_signal.shape[0], end_index)
                    
                    target_audio_mask[start_index:end_index] = True
                
        # Run optimized array modification via memory-bound C-backend vector loops
        attenuated_signal = np.where(
            target_audio_mask, 
            audio_signal, 
            audio_signal * self.attenuation_factor
        )
        
        # Cast clean float metrics back to original int16 header state
        attenuated_signal = attenuated_signal.astype(audio_signal.dtype)
        wavfile.write(output_wav_path, sample_rate, attenuated_signal)
        
        return output_wav_path

    def execute_isolation_pipeline(self, input_wav_path, visual_speech_logs, pyannote_segments, output_dir):
        """
        Orchestration controller for execution tracking.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_wav_path = os.path.join(output_dir, "isolated_target_audio.wav")
        
        target_id, confidence = self.anchor_target_identity(pyannote_segments, visual_speech_logs)
        
        # --- FIX 3: Safe fallback logging for silent targets ---
        if target_id is None:
            print("⚠️ WARNING: Target exhibited zero acoustic lip movement. Attenuating entire audio track.")
            target_id = "TARGET_SILENT"
            confidence = 0.0
            
        self.isolate_voice_channel(input_wav_path, output_wav_path, pyannote_segments, target_id)
        
        manifest_payload = {
            "isolated_audio_source_path": output_wav_path,
            "audio_isolation_metrics": {
                "assigned_target_speaker_id": target_id,
                "cross_modal_correlation_score": float(confidence),
                "attenuation_factor_applied": self.attenuation_factor
            }
        }
        
        print(f"🎯 Voice isolation layer execution complete.")
        print(f"   Target Locked -> {target_id} (Confidence Matrix: {confidence:.5f})")
        return manifest_payload