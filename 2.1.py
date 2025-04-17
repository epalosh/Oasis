#  FALL DETECTION ALGORITHM
#
# By Ethan Palosh, for Oasis
#
#####################################

import cv2
import time
import math
import numpy as np
from ultralytics import YOLO
from collections import deque

# import firebase_admin
# from firebase_admin import credentials, firestore
# from google.cloud.firestore import Increment

# # Initialize Firebase
# cred = credentials.Certificate("/Users/ethanpalosh/Downloads/oasis-196d5-firebase-adminsdk-fbsvc-56d8bea13e.json")
# firebase_admin.initialize_app(cred)

# # Firestore client
# db = firestore.client()

# # Reference to the counter document
# counter_ref = db.collection("counters").document("Fall count")

class FallDetector:
    
    def __init__(
        self,
        model_path: str,
        camera_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        history_len: int = 3,
        retry_count: int = 5,
        retry_delay: float = 0.2,
    ):
        # ===== Loads model and Initializes Camera Input =====
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError("Error: Could not open webcam")

        # ===== Sets resolution for model frames =====
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolution for model processing: {self.width} x {self.height}")

        # ===== Histories for smoothing =====
        # Stores the past 3 frames' datapoints to average during primary logic
        self.hist = {
            'cob_x': deque(maxlen=history_len),
            'cob_y': deque(maxlen=history_len),
            'feet_x': deque(maxlen=history_len),
            'feet_y': deque(maxlen=history_len),
            'area': deque(maxlen=history_len),
            'ratio': deque(maxlen=history_len),
        }

        # COB x,y
        self.prev_mid = (None, None)
        # Area, Ratio
        self.prev_stats = (None, None)

        # Fall stages
        self.stage1 = False
        self.stage2 = False
        self.stage3 = False
        self.fb_fall = False
        self.last_fall_time = 0.0

        # Frame read retries
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        # FPS reporting
        self.fps_count = 0
        self.fps_start = time.time()

    def read_frame(self):
        """Read a frame with retries."""
        for _ in range(self.retry_count):
            ret, frame = self.cap.read()
            if ret:
                return frame
            time.sleep(self.retry_delay)
        raise IOError("Error: Failed to capture image after retries.")

    def update_fps(self):
        """Calculate and print average FPS every 5 seconds."""
        self.fps_count += 1
        now = time.time()
        if now - self.fps_start >= 5.0:
            avg_fps = self.fps_count / 5.0
            print(f"Average FPS over the last 5 seconds: {avg_fps:.2f}")
            self.fps_start = now
            self.fps_count = 0
    
    # def increment_database(self):
    #     # Atomically increment the counter
    #     counter_ref.set({
    #         "count": Increment(1)
    #     }, merge=True)

    def draw_dashboard(self, frame, values):
        """Dashboard drawing logic."""
        dash_w = 600
        dash_color = (30, 30, 30)
        text_color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        lh = 50
        pad = 30
        dash = np.full((frame.shape[0], dash_w, 3), dash_color, dtype=np.uint8)

        # Stage bar
        bar_h, bar_w = 30, dash_w - 2*pad
        stage_w = bar_w // 4
        for i, (flag, label) in enumerate(
            [(self.stage1, 'Stage 1'), (self.stage2, 'Stage 2'), (self.stage3, 'Stage 3'), (self.fb_fall, 'F/B')]
        ):
            x0 = pad + i*stage_w
            color = (0, 255, 0) if flag else (70, 70, 70)
            cv2.rectangle(dash, (x0, 20), (x0+stage_w-5, 20+bar_h), color, -1)
            cv2.putText(
                dash, label, (x0+5, 20+bar_h-8), font, 0.5,
                (0, 0, 0) if flag else (150, 150, 150), 1, cv2.LINE_AA
            )

        y = 20 + bar_h + 30
        for k, v in values.items():
            cv2.putText(dash, f"{k}: {v}", (pad, y), font, scale, text_color, 1, cv2.LINE_AA)
            y += lh

        return np.hstack((frame, dash))

    def annotate(self, frame, values):
        """Title and fall text annotation."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Title
        title = "OASIS Fall Detection v2.1"
        fs, th = 1.5, 2
        (tw, th_), bl = cv2.getTextSize(title, font, fs, th)
        x = frame.shape[1] - tw - 20
        y = th_ + 20
        # black rectangle and title:
        cv2.rectangle(frame, (x-10, y-th_-10), (x+tw+10, y+bl+10), (0, 0, 0), -1)
        cv2.putText(frame, title, (x, y), font, fs, (255,255,255), th, cv2.LINE_AA)

        now = time.time()
        if (self.stage3 or self.fb_fall) and (now - self.last_fall_time < 3.0):
            txt = "FALL DETECTED"
            fs2, th2 = 4, 6
            (tw2, th2_), _ = cv2.getTextSize(txt, font, fs2, th2)
            x2 = (frame.shape[1] - tw2) // 2
            y2 = frame.shape[0] - 100
            cv2.rectangle(
                frame, (x2-10, y2-th2_-10), (x2+tw2+10, y2+10), (0,0,255), cv2.FILLED
            )
            cv2.putText(frame, txt, (x2, y2), font, fs2, (0,0,0), th2, cv2.LINE_AA)
        else:
            # reset transient stages
            self.stage2 = False
            self.stage3 = False
            self.fb_fall = False

        return frame

    def process(self):

        # ===== Key definitions =====
        frame = self.read_frame()
        res = self.model(frame, verbose=False)
        kp = res[0].keypoints
        boxes = res[0].boxes or []

        # ===== Bounding box processing =====
        bbox = max((b for b in boxes if b.conf.item() > 0.5), key=lambda b: b.conf.item(), default=None)
        bx = by = bWidth = bHeight = ba = br = 0
        if bbox:
            x1, y1, x2, y2 = bbox.xyxy[0].tolist()
            bx, by, bWidth, bHeight = map(int, (x1, y1, x2-x1, y2-y1))
            ba = bWidth * bHeight
            br = (bWidth/bHeight) if bHeight else 0
            cv2.rectangle(frame, (bx,by), (bx+bWidth, by+bHeight), (0,255,0), 3)
            if ba: self.hist['area'].append(ba)
            if br: self.hist['ratio'].append(br)

        # ===== Storing box values in bboxData =====
        bboxData = {
            'Box X': bx,
            'Box Y': by,
            'Box W': bWidth,
            'Box H': bHeight,
            'Box Area': ba,
            'Box WH Ratio': round(br,4)
        }

        # ===== Counts number of keypoints =====
        detected = int((kp.data[:,:,2] > 0.5).sum()) if kp is not None else 0
        bboxData['Detected Keypoints'] = detected

        cx = cy = gfx = gfy = 0
        angle = 0.0

        # ===== Primary Detection Logic =====
        if kp is not None and detected >= 4:
            data = kp.data[0]

            # Center of body (hips)
            l_hip, r_hip = data[11], data[12]

            if l_hip[2] > 0.5 and r_hip[2] > 0.5:
                cx, cy = (int(l_hip[0]+r_hip[0])//2, int(l_hip[1]+r_hip[1])//2)
                self.hist['cob_x'].append(cx)
                self.hist['cob_y'].append(cy)
                prev_cx, prev_cy = self.prev_mid
                avg_cx = sum(self.hist['cob_x'])/len(self.hist['cob_x'])
                avg_cy = sum(self.hist['cob_y'])/len(self.hist['cob_y'])

                sb_area = sum(self.hist['area'])/len(self.hist['area']) if self.hist['area'] else None
                sb_ratio = sum(self.hist['ratio'])/len(self.hist['ratio']) if self.hist['ratio'] else None

                # Stage 1 & forward/back
                tnow = time.time()
                if prev_cx is not None and (tnow - self.last_fall_time) > 0.25:
                    self.stage1 = (avg_cy - prev_cy) > 20
                    # if(avg_cy - prev_cy > 0):
                    #     print(str(avg_cy - prev_cy))

                    #F/B logic
                    if self.stage1 and sb_area and sb_ratio:
                        if sb_area < 0.83 * self.prev_stats[0] and sb_ratio > 1.1 * self.prev_stats[1]:
                            if (time.time() - self.last_fall_time) > 3.0:    
                                self.fb_fall = True
                                print("FALL DETECTED")
                                self.last_fall_time = tnow

                self.prev_mid = (avg_cx, avg_cy)
                if sb_area is not None and sb_ratio is not None:
                    self.prev_stats = (sb_area, sb_ratio)

                # Feet midpoint & angle
                l_foot, r_foot, nose = data[15], data[16], data[0]
                if nose[2] > 0.5 and l_foot[2] > 0.5 and r_foot[2] > 0.5:
                    gfx, gfy = (int(l_foot[0]+r_foot[0])//2, int(l_foot[1]+r_foot[1])//2)
                    self.hist['feet_x'].append(gfx)
                    self.hist['feet_y'].append(gfy)
                    avg_fx = sum(self.hist['feet_x'])/len(self.hist['feet_x'])
                    avg_fy = sum(self.hist['feet_y'])/len(self.hist['feet_y'])

                    # Calc midline angle, and draw line
                    angle = math.degrees(math.atan2(avg_fy - int(nose[1]), avg_fx - int(nose[0])))
                    cv2.line(frame, (int(nose[0]), int(nose[1])), (int(avg_fx), int(avg_fy)), (255,0,0), 3)

                    # Stage 2 & 3
                    if self.stage1 and (abs(angle) < 65 or abs(angle) > 115):
                        self.stage2 = True
                        if bWidth and bHeight and (bWidth/bHeight) > 1.3:
                            if (time.time() - self.last_fall_time) > 3.0:
                                self.stage3 = True
                                print("FALL DETECTED")
                                self.last_fall_time = time.time()
                    else:
                        self.stage2 = False

                # Fallback when only nose but ensure prev_mid exists
                elif nose[2] > 0.5 and self.prev_mid[0] is not None:
                    nx, ny = int(data[0][0]), int(data[0][1]) # clean
                    avg_cx, avg_cy = self.prev_mid
                    cv2.line(frame, (nx, ny), (int(avg_cx), int(avg_cy)), (255,0,0), 3)
                    angle = math.degrees(math.atan2(avg_cy - ny, avg_cx - nx))
                    if self.stage1 and (abs(angle) < 65 or abs(angle) > 115):
                        self.stage2 = True
                        if bWidth and bHeight and (bWidth/bHeight) > 1.3:
                            if (time.time() - self.last_fall_time) > 3.0:
                                self.stage3 = True
                                print("FALL DETECTED")
                                self.last_fall_time = time.time()
                    else:
                        self.stage2 = False

            # Populate smoothed values
            bboxData.update({
                'Avg Midpoint X': int(self.prev_mid[0]) if self.prev_mid[0] is not None else 0,
                'Avg Midpoint Y': int(self.prev_mid[1]) if self.prev_mid[1] is not None else 0,
                'Feet Midpoint X': int(gfx),
                'Feet Midpoint Y': int(gfy),
                'Midline': f"{angle:.1f} degrees"
            })

        # ===== Log Fall Status =====
        bboxData['Fall Detected'] = self.stage3
        bboxData['Forward/Back Fall Detected'] = self.fb_fall

        # ==== Plot and Return Annotated View =====
        frame = res[0].plot(boxes=False, masks=True, probs=False, labels=False, line_width=4, font_size=1)
        frame = self.annotate(frame, bboxData)
        frame = self.draw_dashboard(frame, bboxData)
        self.update_fps()
        return frame

    def run(self):
        try:
            while True:
                frame = self.process()
                cv2.imshow("Fall Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    FallDetector("yolo11n-pose.pt", width=1920, height=1080).run()

#I would like to take this logic and make it work for multiple people in the frame. If there are several detected humans, I want their falls to be detected independently (it's ok to use the same variables for fall indication). I need it to be resilient to people gfoing in and out of frame, and I dont know how I would go about that.