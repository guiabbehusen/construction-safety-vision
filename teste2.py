import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("yolov8m.pt")

path = "C:/Users/User/Documents/GitHub/construction-safety-vision/videos/video4.mp4"
cap = cv2.VideoCapture(path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps) * 2

prev_centers = {}
frame_count = {}
motion_threshold = 60
min_frames_for_bruise = 4

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('C:/Users/User/Documents/GitHub/construction-safety-vision/resultados2/resultado_video4_m.pt.mp4',
                      fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes
    current_centers = {}

    for i, box in enumerate(boxes):
        cls = int(box.cls[0].item())
        if cls != 0:
            continue

        x1, y1, x2, y2 = box.xyxy[0].clone()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        person_id = i
        if person_id in prev_centers:
            prev_cx, prev_cy = prev_centers[person_id]
            dist = np.linalg.norm([cx - prev_cx, cy - prev_cy])

            if dist > motion_threshold:
                frame_count[person_id] = frame_count.get(person_id, 0) + 1

                if frame_count[person_id] >= min_frames_for_bruise:
                    cv2.putText(frame, "ACIDENTE", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_count[person_id] = 0

        current_centers[person_id] = (cx, cy)

    prev_centers = current_centers

    out.write(frame)
    cv2.imshow("Detecção", frame)

    if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
