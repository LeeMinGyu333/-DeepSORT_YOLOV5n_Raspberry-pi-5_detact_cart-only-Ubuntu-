# -DeepSORT_YOLOV5n_Raspberry-pi-5_detact_cart-only-Ubuntu-



# Tracking with YOLOv5n + DeepSORT in Ubuntu

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import cv2
import torch
import random
from deep_sort_realtime.deepsort_tracker import DeepSort  # pip install deep_sort_realtime

# YOLO model loaded
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best2.pt', device='cpu', force_reload=True)
print("✅ YOLOv5n model loaded")

#DeepSort init
tracker = DeepSort(max_age=30)

# color palette(max 50)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(50)]

# open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ cannot open webcam.")
    exit()

detection_threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ reading frame failed.")
        break

    # YOLOv5 input: BGR image 
    results = model(frame)

    detections = []
    for *box, conf, cls_id in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        label = int(cls_id)

        # COCO class 0 = person
        if conf > detection_threshold and label == 0:
            detections.append(([x1, y1, x2 - x1, y2 - y1], float(conf), 'person'))

    # DeepSORT tracking
    tracks = tracker.update_tracks(detections, frame=frame)

    # visualization
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        color = colors[int(track_id) % len(colors)]

        cv2.rectangle(frame, (l, t), (r, b), color, 2)
        cv2.putText(frame, f"ID: {track_id}", (l, t - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # screen output
    cv2.imshow('YOLOv5n + DeepSORT Tracking', frame)

    # quit: press 'q' 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# arrange resources
cap.release()
cv2.destroyAllWindows()
