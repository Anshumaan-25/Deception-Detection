import cv2
import torch
import numpy as np
import logging
from typing import Dict, Any

# Absolute imports mapping to our newly created detector modules
from detectors.face_detector import FaceDetector
from detectors.landmark_detector import LandmarkDetector
from detectors.unified_detector import UnifiedBehaviorDetector

class OpenFaceExtractor:
    def __init__(self, legacy_weights_dir: str, device: str = "cuda"):
        """
        The Main Gateway API.
        Orchestrates RetinaFace, STAR, and the MLT Unified Behavior networks.
        """
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger("OpenFace_API_Gateway")
        self.device = torch.device(device)
        self.mlt_stream = torch.cuda.Stream(device=self.device)
        
        self.logger.info("Booting OpenFace 3.0 Modular Pipeline...")
        
        # Instantiate the decoupled engines
        # Note: Filenames should match the actual legacy weights in your directory
        self.face_detector = FaceDetector(
            weights_path=f"{legacy_weights_dir}/mobilenet0.25_Final.pth", 
            device=device
        )
        self.landmark_detector = LandmarkDetector(
            weights_path=f"{legacy_weights_dir}/WFLW_STARLoss_NME_4_02_FR_2_32_AUC_0_605.pkl", 
            device=device
        )
        self.behavior_detector = UnifiedBehaviorDetector(
            # Assuming the MLT weights file. Adjust if the filename differs in the legacy repo.
            weights_path=f"{legacy_weights_dir}/MLT_model.pth", 
            device=device
        )
        self.logger.info("All CUDA engines loaded and synchronized.")

    def process_frame(self, frame: np.ndarray, frame_id: int = 0, timestamp_ms: float = 0.0) -> Dict[str, Any]:
        """
        Full inference pipeline for a single frame.
        """
        # --- Input Validation Guard ---
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            self.logger.warning("Invalid or empty frame received, skipping.")
            return {"frame_id": frame_id, "timestamp_ms": timestamp_ms, "faces": []}
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.logger.warning("Frame must be a 3-channel image array, skipping.")
            return {"frame_id": frame_id, "timestamp_ms": timestamp_ms, "faces": []}

        # --- 1. Face Detection Preprocessing ---
        img_raw = np.float32(frame)
        im_height, im_width = img_raw.shape[:2]
        
        # RetinaFace specific mean subtraction
        img = img_raw - (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img).unsqueeze(0).pin_memory().to(self.device, non_blocking=True)

        # --- 2. RetinaFace Forward Pass ---
        dets = self.face_detector.detect(img_tensor, (im_height, im_width))
        
        if dets.size(0) == 0:
            return {"frame_id": frame_id, "timestamp_ms": timestamp_ms, "faces": []}

        faces_data = []
        batched_crops = []
        valid_bboxes = []
        landmarks_list = []

        # --- 3. Landmark & Behavior Cropping ---
        for i in range(dets.size(0)):
            bbox = dets[i, :4].cpu().numpy()
            conf = float(dets[i, 4].cpu().item())
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clamp coordinates strictly within frame dimensions
            x1 = max(0, min(x1, im_width - 1))
            y1 = max(0, min(y1, im_height - 1))
            x2 = max(0, min(x2, im_width))
            y2 = max(0, min(y2, im_height))
            
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            
            # STAR Landmark Pass
            landmarks = self.landmark_detector.detect(frame, bbox)
            
            # MLT Behavior Crop & Preprocess
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            
            # ImageNet Normalization for EfficientNet Backbone
            face_tensor = face_resized.astype(np.float32) / 255.0
            face_tensor = (face_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            face_tensor = face_tensor.transpose(2, 0, 1)
            
            face_tensor_th = torch.from_numpy(face_tensor).pin_memory()
            batched_crops.append(face_tensor_th)
            valid_bboxes.append((bbox, conf))
            landmarks_list.append(landmarks)

        if not batched_crops:
            return {"frame_id": frame_id, "timestamp_ms": timestamp_ms, "faces": []}

        # --- 4. Unified Behavior Forward Pass (Batched) ---
        with torch.cuda.stream(self.mlt_stream):
            batch_tensor = torch.stack(batched_crops).to(self.device, non_blocking=True)
            behavior_results = self.behavior_detector.analyze(batch_tensor)

        # --- 5. Construct JSON-Ready Output ---
        for idx, behavior in enumerate(behavior_results):
            bbox, conf = valid_bboxes[idx]
            landmarks = landmarks_list[idx]
            
            faces_data.append({
                "face_id": idx, # Placeholder for downstream deep tracking ID
                "bbox": bbox.tolist(),
                "confidence": conf,
                "landmarks": landmarks.tolist(),
                "emotion": behavior["emotion"],
                "gaze_3d": behavior["gaze_3d"],
                "action_units": behavior["action_units"]
            })

        return {
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "faces": faces_data
        }

    def process_batch(self, crops_list: list, frame_ids: list, timestamps_list: list) -> list:
        """
        Full inference pipeline for a batch of frames/crops.
        Processes face detection, landmark detection, and runs a single
        heavy batched MLT behavior analysis forward pass on GPU.

        Args:
            crops_list: List of numpy person crops (can contain None for tracking misses)
            frame_ids: List of corresponding frame IDs
            timestamps_list: List of corresponding timestamps in ms
        Returns:
            List of dictionaries matching the input size, with tracking misses filled with default/NaN.
        """
        if not crops_list:
            return []

        B = len(crops_list)
        results = [None] * B

        # Staging for batch inference
        valid_indices = []
        batched_crops = []
        valid_bboxes = []
        landmarks_list = []

        # Step 1: Sequential/Loop face detection & landmark extraction per valid crop.
        # Person crops are typically small, so running RetinaFace + STAR per valid crop
        # sequentially on GPU is extremely fast, while the heavy EfficientNet-B0 MLT
        # backbone is fully stacked and executed in a single batched pass on GPU.
        for idx, crop in enumerate(crops_list):
            fid = frame_ids[idx]
            ts = timestamps_list[idx]

            # Default empty result template for this frame position
            results[idx] = {
                "frame_id": fid,
                "timestamp_ms": ts,
                "faces": []
            }

            if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
                continue

            # RetinaFace Preprocessing
            img_raw = np.float32(crop)
            im_height, im_width = img_raw.shape[:2]
            img = img_raw - (104, 117, 123)
            img = img.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img).unsqueeze(0).pin_memory().to(self.device, non_blocking=True)

            # RetinaFace Face Detection
            dets = self.face_detector.detect(img_tensor, (im_height, im_width))
            if dets.size(0) == 0:
                continue

            # We lock onto the single primary face in the person crop (highest conf)
            bbox = dets[0, :4].cpu().numpy()
            conf = float(dets[0, 4].cpu().item())

            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, min(x1, im_width - 1))
            y1 = max(0, min(y1, im_height - 1))
            x2 = max(0, min(x2, im_width))
            y2 = max(0, min(y2, im_height))

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            # STAR Landmark Pass
            landmarks = self.landmark_detector.detect(crop, bbox)

            # MLT Behavior Crop, Resize & Normalize
            face_crop = crop[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))

            face_tensor = face_resized.astype(np.float32) / 255.0
            face_tensor = (face_tensor - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            face_tensor = face_tensor.transpose(2, 0, 1)

            # Stage for batched forward pass
            valid_indices.append(idx)
            face_tensor_th = torch.from_numpy(face_tensor).pin_memory()
            batched_crops.append(face_tensor_th)
            valid_bboxes.append((bbox, conf))
            landmarks_list.append(landmarks)

        # Step 2: Unified Batched MLT Forward Pass (If any valid faces were detected)
        if batched_crops:
            with torch.cuda.stream(self.mlt_stream):
                batch_tensor = torch.stack(batched_crops).to(self.device, non_blocking=True)
                # Run the single heavy batched forward pass on GPU
                behavior_results = self.behavior_detector.analyze(batch_tensor)

            # Step 3: Unpack results and assign them back to their correct chronological indexes
            for i, idx in enumerate(valid_indices):
                behavior = behavior_results[i]
                bbox, conf = valid_bboxes[i]
                landmarks = landmarks_list[i]

                results[idx]["faces"] = [{
                    "face_id": 0,
                    "bbox": bbox.tolist(),
                    "confidence": conf,
                    "landmarks": landmarks.tolist(),
                    "emotion": behavior["emotion"],
                    "gaze_3d": behavior["gaze_3d"],
                    "action_units": behavior["action_units"]
                }]

        return results

