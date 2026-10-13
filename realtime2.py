
import cv2
import time
import threading
import queue
import pyttsx3
from ultralytics import YOLO

MODEL_PATH = "best.pt"
CONFIDENCE = 0.45
COOLDOWN = 5
ABSENCE_TIME = 2

model = YOLO(MODEL_PATH)

# ---------------- SPEECH ----------------
speech_queue = queue.Queue()

def speech_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        engine.setProperty("volume", 1.0)

        while True:
            message = speech_queue.get()

            if message is None:
                speech_queue.task_done()
                break

            try:
                print("TTS:", message)
                engine.say(message)
                engine.runAndWait()
            except Exception as e:
                print("TTS ERROR:", e)
            finally:
                speech_queue.task_done()

        engine.stop()

    except Exception as e:
        print("Speech engine initialization failed:", e)

threading.Thread(target=speech_worker, daemon=True).start()


def speak(message):
    print("QUEUED:", message)
    speech_queue.put(message)


# ---------------- POSITION ----------------
def get_zone(x, width):
    ratio = x / width

    if ratio < 1 / 3:
        return "left"
    elif ratio > 2 / 3:
        return "right"
    return "center"


def make_message(name, zone):
    if zone == "center":
        return f"{name} ahead"
    return f"{name} on your {zone}"


# State per detected class
state = {}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")

print("Navigation started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        height, width = frame.shape[:2]

        # Zone boundaries
        cv2.line(frame, (width // 3, 0),
                 (width // 3, height), (255, 255, 0), 2)
        cv2.line(frame, (2 * width // 3, 0),
                 (2 * width // 3, height), (255, 255, 0), 2)

        result = model.predict(
            frame, imgsz=640, conf=CONFIDENCE, verbose=False
        )[0]

        # Keep strongest detection per class
        visible = {}

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            name = model.names[class_id]
            confidence = float(box.conf[0].item())

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            center_x = (x1 + x2) // 2
            zone = get_zone(center_x, width)

            if name not in visible or confidence > visible[name]["confidence"]:
                visible[name] = {
                    "zone": zone,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2)
                }

        # Announcements and display
        for name, detection in visible.items():
            zone = detection["zone"]
            x1, y1, x2, y2 = detection["box"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, f"{name} - {zone}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

            previous = state.get(name)
            announce = False

            if previous is None:
                # First detection
                announce = True

            elif now - previous["last_seen"] >= ABSENCE_TIME:
                # Object disappeared and returned
                announce = True

            elif (
                zone != previous["zone"]
                and now - previous["last_spoken"] >= COOLDOWN
            ):
                # Position changed after cooldown
                announce = True

            if announce:
                speak(make_message(name, zone))
                state[name] = {
                    "zone": zone,
                    "last_seen": now,
                    "last_spoken": now
                }
            else:
                # Continue tracking presence
                state[name]["last_seen"] = now
                state[name]["zone"] = zone

        # Remove objects that have been absent long enough.
        # Their return will trigger a new announcement.
        for name in list(state):
            if name not in visible:
                if now - state[name]["last_seen"] >= ABSENCE_TIME:
                    del state[name]

        cv2.imshow("Indoor Navigation", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    speech_queue.join()
    speech_queue.put(None)
    print("Navigation stopped.")
