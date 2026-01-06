import cv2
from ultralytics import YOLO

# Load YOLOv8 model (pretrained)
model = YOLO("yolov8n.pt")

# Open camera (0 = default webcam)
cap = cv2.VideoCapture(0)

# Weapon classes (COCO does NOT include weapons by default)
WEAPON_KEYWORDS = ["knife", "gun", "pistol", "rifle"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            confidence = float(box.conf[0])

            if label.lower() in WEAPON_KEYWORDS and confidence > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    f"WEAPON: {label} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

    cv2.imshow("Weapon Detection Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
