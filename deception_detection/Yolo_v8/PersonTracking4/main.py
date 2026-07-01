import cv2
import argparse
import torch

from src.detector import PersonDetector
from src.click_selector import ClickSelector
from src.tracker import TargetTracker
from src.face_lock import FaceLock


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True)
    return p.parse_args()


def main():

    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(args.video)

    detector = PersonDetector(args.model, device)
    selector = ClickSelector()
    tracker = TargetTracker()
    facelock = FaceLock()

    window = "Face Identity Lock Tracking"
    cv2.namedWindow(window)
    selector.register(window)

    locked_bbox = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect_and_track(frame)

        # ---------------------------------
        # USER CLICK → LOCK FACE
        # ---------------------------------
        if facelock.target_embedding is None:

            tid = selector.select_target(detections)

            if tid is not None:
                for d in detections:
                    if d["id"] == tid:
                        facelock.lock(frame, d["bbox"])

        # ---------------------------------
        # MATCH IDENTITY
        # ---------------------------------
        else:
            locked_bbox = facelock.match(frame, detections)

        frame = tracker.draw(frame, detections, locked_bbox)

        cv2.imshow(window, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

