import cv2
from ultralytics import YOLO
import time
import math
import numpy as np
from collections import deque

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Resize the frames to 720p
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Resolution: {int(width)} x {int(height)}")

prev_midpoint_x = None
prev_midpoint_y = None
prev_bbox_area = None
prev_bbox_ratio = None

stage_1 = False
stage_2 = False
stage_3 = False # likely detected a fall
forwardOrBackward_fall= False

prev_time = time.time()
fall_detected_time = 0

COM_x_history = deque(maxlen=3)
COM_y_history = deque(maxlen=3)
feet_midpoint_x_history = deque(maxlen=3)
feet_midpoint_y_history = deque(maxlen=3)

box_area_history = deque(maxlen=3)
box_ratio_history = deque(maxlen=3)

def draw_dashboard(frame, values_dict, stage_1, stage_2, stage_3, forwardOrBackward_fall):
    dashboard_width = 600
    dashboard_color = (30, 30, 30)
    text_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    line_height = 50
    padding = 30

    dash = np.full((frame.shape[0], dashboard_width, 3), dashboard_color, dtype=np.uint8)

    # === Draw Fall Detection Stage Bar ===
    bar_top = 20
    bar_left = padding
    bar_height = 30
    bar_width = dashboard_width - 2 * padding
    stage_width = bar_width // 4

    stages = [(stage_1, "Stage 1"), (stage_2, "Stage 2"), (stage_3, "Stage 3"), (forwardOrBackward_fall, "F/B")]
    for i, (active, label) in enumerate(stages):
        left = bar_left + i * stage_width
        color = (0, 255, 0) if active else (70, 70, 70)
        cv2.rectangle(dash, (left, bar_top), (left + stage_width - 5, bar_top + bar_height), color, -1)
        cv2.putText(dash, label, (left + 5, bar_top + bar_height - 8), font, 0.5, (0, 0, 0) if active else (150, 150, 150), 1, cv2.LINE_AA)

    # === Add other data to dashboard ===
    y_offset = bar_top + bar_height + 30
    for label, value in values_dict.items():
        line = f"{label}: {value}"
        cv2.putText(dash, line, (padding, y_offset), font, font_scale, text_color, 1, cv2.LINE_AA)
        y_offset += line_height

    return np.hstack((frame, dash))
    

bbox_ratio = 0 
bbox_area = 0
bbox_x = 0
bbox_y = 0
bbox_w = 0
bbox_h = 0

# FPS Calculation Variables
frame_count = 0
fps_start_time = time.time()

while True:
    # ret, frame = cap.read()
    # if not ret:
    #     print("Error: Failed to capture image")
    #     break

    max_retries = 5
    retry_delay = 0.2  # seconds

    for attempt in range(max_retries):
        ret, frame = cap.read()
        if ret:
            break
        else:
            print(f"Warning: Failed to capture image. Retrying... ({attempt + 1}/{max_retries})")
            time.sleep(retry_delay)
    else:
        print("Error: Failed to capture image after multiple retries.")
        break

    results = model(frame, verbose=False)
    keypoints = results[0].keypoints

    # Defaults
    avg_midpoint_x = avg_midpoint_y = 0
    avg_feet_midpoint_x = avg_feet_midpoint_y = 0
    angle_deg = 0

    if keypoints is not None:
        keypoints_data = keypoints.data
        detected_keypoints = sum(1 for i in range(keypoints_data.shape[1]) if keypoints_data[0, i, 2] > 0.5)

        bbox_x = bbox_y = bbox_w = bbox_h = 0  # defaults

        if results[0].boxes is not None and len(results[0].boxes) > 0:

            # Loop through all boxes and find the one with the highest confidence > 0.5
            selected_box = None
            max_conf = 0.5  # Threshold

            for i, box in enumerate(results[0].boxes):
                conf = box.conf.item()
                if conf > max_conf:
                    selected_box = box
                    max_conf = conf

            if selected_box is not None:
                x1, y1, x2, y2 = selected_box.xyxy[0]
                bbox_x = int(x1.item())
                bbox_y = int(y1.item())
                bbox_w = int((x2 - x1).item())
                bbox_h = int((y2 - y1).item())

                bbox_area = bbox_w*bbox_h
                if bbox_h != 0: bbox_ratio = bbox_w/bbox_h 
                else: bbox_ratio = 0 

                if bbox_area != 0:
                    box_area_history.append(bbox_area)
                if bbox_ratio != 0:
                    box_ratio_history.append(bbox_ratio)

                # ✅ Draw green bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)


        if detected_keypoints >= 4:
            left_hip = keypoints_data[0, 11]
            right_hip = keypoints_data[0, 12]

            if left_hip[2] > 0.5 and right_hip[2] > 0.5:
                left_hip_x, left_hip_y = int(left_hip[0]), int(left_hip[1])
                right_hip_x, right_hip_y = int(right_hip[0]), int(right_hip[1])

                midpoint_x = (left_hip_x + right_hip_x) / 2
                midpoint_y = (left_hip_y + right_hip_y) / 2

                COM_x_history.append(midpoint_x)
                COM_y_history.append(midpoint_y)

                avg_midpoint_x = sum(COM_x_history) / len(COM_x_history)
                avg_midpoint_y = sum(COM_y_history) / len(COM_y_history)

                avg_bbx_area = sum(box_area_history) / len(box_area_history)
                avg_bbx_ratio = sum(box_ratio_history) / len(box_ratio_history)

                current_time = time.time()
                if current_time - prev_time > 0.25:
                    prev_time = current_time
                    if prev_midpoint_x is not None and prev_midpoint_y is not None:
                        midpoint_fall_y = avg_midpoint_y - prev_midpoint_y
                        stage_1 = midpoint_fall_y > 30
                        if stage_1 and prev_bbox_area is not None and prev_bbox_ratio is not None:
                            if (avg_bbx_area < .83 * prev_bbox_area) and (avg_bbx_ratio > 1.1 * prev_bbox_ratio):
                                forwardOrBackward_fall = True
                                print("FALL DETECTED")
                                fall_detected_time = time.time()

                        #here
                    prev_midpoint_x, prev_midpoint_y = avg_midpoint_x, avg_midpoint_y
                    prev_bbox_area, prev_bbox_ratio = avg_bbx_area, avg_bbx_ratio

                nose = keypoints_data[0, 0]
                left_foot = keypoints_data[0, 15]
                right_foot = keypoints_data[0, 16]

                left_foot_x, left_foot_y = int(left_foot[0]), int(left_foot[1])
                right_foot_x, right_foot_y = int(right_foot[0]), int(right_foot[1])

                feet_midpoint_x = (left_foot_x + right_foot_x) // 2
                feet_midpoint_y = (left_foot_y + right_foot_y) // 2

                feet_midpoint_x_history.append(feet_midpoint_x)
                feet_midpoint_y_history.append(feet_midpoint_y)

                avg_feet_midpoint_x = sum(feet_midpoint_x_history) / len(feet_midpoint_x_history)
                avg_feet_midpoint_y = sum(feet_midpoint_y_history) / len(feet_midpoint_y_history)

                

                if nose[2] > 0.5 and left_foot[2] > 0.5 and right_foot[2] > 0.5:
                    # cv2.circle(frame, (int(avg_feet_midpoint_x), int(avg_feet_midpoint_y)), 50, (255, 255, 255), -1)

                    nose_x, nose_y = int(nose[0]), int(nose[1])
                    cv2.line(frame, (nose_x, nose_y), (int(avg_feet_midpoint_x), int(avg_feet_midpoint_y)), (255, 0, 0), 3)

                    dx = avg_feet_midpoint_x - nose_x
                    dy = avg_feet_midpoint_y - nose_y
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)

                    if stage_1 and (abs(angle_deg) < 65 or abs(angle_deg) > 115):
                        stage_2 = True
                        if(bbox_h != 0):
                            if (bbox_w / bbox_h) > 1.3:
                                stage_3 = True
                                print("FALL DETECTED")
                                fall_detected_time = time.time()
                    else:
                        stage_2 = False

                elif nose[2] > 0.5:
                # if nose[2] > 0.5:
                    nose_x, nose_y = int(nose[0]), int(nose[1])
                    cv2.line(frame, (nose_x, nose_y), (int(avg_midpoint_x), int(avg_midpoint_y)), (255, 0, 0), 3)
                    dx = avg_midpoint_x - nose_x
                    dy = avg_midpoint_y - nose_y
                    angle_rad = math.atan2(dy, dx)
                    angle_deg = math.degrees(angle_rad)
                    if stage_1 and (abs(angle_deg) < 65 or abs(angle_deg) > 115):
                        stage_2 = True
                        if(bbox_h != 0):
                            if (bbox_w / bbox_h) > 1.3:
                                stage_3 = True
                                print("FALL DETECTED")
                                fall_detected_time = time.time()
                    else:
                        stage_2 = False


        frame = results[0].plot(boxes=False, masks=True, probs=False, labels=False, line_width=4, font_size=1)

    title_text = "OASIS Fall Detection v1.4"
    font_scale = 1.5
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size with the CORRECT font scale
    (font_width, font_height), baseline = cv2.getTextSize(title_text, font, font_scale, thickness)

    # Set position (top-right corner with padding)
    x_text = frame.shape[1] - font_width - 20
    y_text = font_height + 20

    # Draw black rectangle background
    cv2.rectangle(
        frame,
        (x_text - 10, y_text - font_height - 10),  # top-left corner
        (x_text + font_width + 10, y_text + baseline + 10),  # bottom-right corner
        (0, 0, 0),
        thickness=-1
    )

    # Draw white title text
    cv2.putText(
        frame,
        title_text,
        (x_text, y_text),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    if stage_1:
        fall_color = (255, 69, 0)
        fall_text = "STAGE 1 DETECT"
    else:
        fall_color = (0, 255, 0)
        fall_text = "No Fall Detected"

    if (stage_3 or forwardOrBackward_fall) and (time.time() - fall_detected_time < 3):
        fall_display_text = "FALL DETECTED"
        font_scale = 4
        thickness = 6
        text_size, _ = cv2.getTextSize(fall_display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 100
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, fall_display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    else:
        stage_3 = False
        forwardOrBackward_fall = False

    # 🧾 Collect all variables to display
    dashboard_data = {
        "Avg Midpoint X": f"{int(avg_midpoint_x)}",
        "Avg Midpoint Y": f"{int(avg_midpoint_y)}",
        "Feet Midpoint X": f"{int(avg_feet_midpoint_x)}",
        "Feet Midpoint Y": f"{int(avg_feet_midpoint_y)}",
        "Midline": f"{angle_deg:.1f} degrees",
        "Detected Keypoints": str(detected_keypoints),
        "Box X": f"{bbox_x}",
        "Box Y": f"{bbox_y}",
        "Box W": f"{bbox_w}",
        "Box H": f"{bbox_h}",
        "Box WH Ratio": f"{round(bbox_ratio, 4)}",
        "Box Area": f"{bbox_area}",
        "Time": f"{time.strftime('%H:%M:%S')}",
        "Fall Detected": str(stage_3),
        "Forward/Back Fall Detected": str(forwardOrBackward_fall),
    }


    # FPS Calculation: Increment frame count
    frame_count += 1

    # Every 5 seconds, calculate and print the average FPS
    current_time = time.time()
    if current_time - fps_start_time >= 5:
        avg_fps = frame_count / 5.0
        print(f"Average FPS over the last 5 seconds: {avg_fps:.2f}")

        # Reset the FPS counter and start time for the next interval
        frame_count = 0
        fps_start_time = current_time


    # frame_with_dashboard = draw_dashboard(frame, dashboard_data, stage_1, stage_2, stage_3, forwardOrBackward_fall)

    # cv2.imshow("Fall Detection + Dashboard", frame_with_dashboard)
    # cv2.imshow("Fall Detection + Dashboard", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
