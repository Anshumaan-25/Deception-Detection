import cv2

class ClickSelector:

    def __init__(self):
        self.clicked_point = None
        self.target_id = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x,y)

    def register(self, window_name):
        cv2.setMouseCallback(window_name, self.mouse_callback)

    def select_target(self, detections):

        if self.clicked_point is None:
            return self.target_id

        px,py = self.clicked_point

        for det in detections:
            x1,y1,x2,y2 = det["bbox"]

            if x1 <= px <= x2 and y1 <= py <= y2:
                self.target_id = det["id"]
                print(f"\n Locked Target ID = {self.target_id}")

        self.clicked_point = None
        return self.target_id

