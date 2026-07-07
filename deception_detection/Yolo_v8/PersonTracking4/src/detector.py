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

        # Hand the list of BGR numpy frames straight to Ultralytics. An earlier
        # version stacked these into a pinned GPU tensor and passed a LIST OF
        # torch tensors, but ultralytics>=8.1 rejects Tensor list-elements as a
        # prediction source ("type Tensor is not a supported source type"). A
        # list of HWC numpy arrays is a supported batch source, and ultralytics
        # performs the pinned host->device copy internally.
        # Default CUDA stream. A custom torch.cuda.Stream here collided with
        # onnxruntime's CUDA EP (InsightFace, now GPU) — "operation not
        # permitted when stream is capturing". The stream was only a throughput
        # micro-opt; correctness-first runs use the default stream.
        results = self.model.track(
            frames_list,
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


