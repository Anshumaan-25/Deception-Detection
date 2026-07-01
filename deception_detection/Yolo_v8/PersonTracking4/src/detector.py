import torch
from ultralytics import YOLO

class PersonDetector:

    def __init__(self, model_path, device='cuda'):
        self.model = YOLO(model_path)
        self.device = device
        self.model.to(device)
        self.yolo_stream = torch.cuda.Stream(device=self.device)
        print(f'Person Detector model initialized on {self.device}')

    def detect_and_track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            classes=[0],   # person only
            conf=0.4,
            verbose=False
        )

        detections = []

        if results[0].boxes.id is None:
            return detections

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, tid, conf in zip(boxes, ids, confs):

            x1,y1,x2,y2 = map(int, box)

            detections.append({
                "id": int(tid),
                "bbox": (x1,y1,x2,y2),
                "conf": float(conf)
            })

        return detections

    def detect_and_track_batch(self, frames_list):
        """
        Runs tracking in a parallel GPU batch pass utilizing pinned memory and non-blocking DMA.
        """
        if not frames_list:
            return []

        # Enforce pinned host memory allocation and non-blocking PCIe migration
        import numpy as np
        # Convert list of BGR numpy frames to a single stacked tensor (B, H, W, C)
        batch_numpy = np.stack(frames_list)
        # Allocate pinned memory buffer directly
        batch_tensor = torch.from_numpy(batch_numpy).pin_memory()
        # Non-blocking copy to VRAM across PCIe
        batch_tensor_gpu = batch_tensor.to(self.device, non_blocking=True)
        # Convert to list of tensors for Ultralytics YOLO track API compatibility
        batch_tensors_list = [batch_tensor_gpu[i] for i in range(batch_tensor_gpu.shape[0])]

        with torch.cuda.stream(self.yolo_stream):
            results = self.model.track(
                batch_tensors_list,
                persist=True,
                classes=[0],   # person only
                conf=0.4,
                verbose=False
            )

        batch_detections = []
        for r in results:
            detections = []
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                for box, tid, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        "id": int(tid),
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(conf)
                    })
            batch_detections.append(detections)

        return batch_detections


