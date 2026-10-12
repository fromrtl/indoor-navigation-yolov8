
import cv2
from ultralytics import YOLO

# Load the newly trained model
model = YOLO("best_door_finetune.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Detect objects
    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.35,
        verbose=False
    )

    # Draw bounding boxes and labels
    annotated_frame = results[0].plot()

    cv2.imshow("Indoor Navigation - YOLOv8", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
