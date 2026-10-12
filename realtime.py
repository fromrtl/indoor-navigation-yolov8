from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Open laptop webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    height, width = frame.shape[:2]

    # YOLO detection
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    print("\n--- DETECTIONS ---")

    if len(results[0].boxes) == 0:
        print("No objects detected")

    # Process every detection
    for box in results[0].boxes:

        # Bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        # Confidence
        confidence = float(box.conf[0])

        # Class
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Bounding box center
        center_x = (x1 + x2) / 2

        # Determine position
        if center_x < width / 3:
            position = "LEFT"

        elif center_x < 2 * width / 3:
            position = "CENTER"

        else:
            position = "RIGHT"

        # Print to terminal
        print(
            f"{class_name} | "
            f"Confidence: {confidence:.2f} | "
            f"Position: {position}"
        )

    # Draw YOLO boxes on webcam
    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Navigation", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()