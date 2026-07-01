import cv2

class TargetTracker:

    def draw(self, frame, detections, locked_bbox):

        for det in detections:

            x1,y1,x2,y2 = det["bbox"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        if locked_bbox is not None:
            x1,y1,x2,y2 = locked_bbox
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
            cv2.putText(frame,"FACE LOCK",
                        (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,0,255),2)

        return frame

