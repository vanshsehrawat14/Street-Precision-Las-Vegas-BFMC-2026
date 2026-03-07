import cv2
import base64
import requests
import numpy as np

API_KEY = "WfannhlmdFjwemdK3rzY"
MODEL_ID = "bfmc-6btkg/3"
API_URL = "https://detect.roboflow.com/bfmc-6btkg/3"

STOP_CLASSES = {"stop_sign", "traffic_light"}
SLOW_CLASSES = {"pedestrian", "crosswalk"}


def encode_frame(frame):
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def run_detection(frame):
    encoded = encode_frame(frame)
    try:
        response = requests.post(
            API_URL,
            params={"api_key": API_KEY},
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("predictions", [])
    except Exception as e:
        print(f"[Detection] API error: {e}")
        return []


def draw_detections(frame, predictions):
    for pred in predictions:
        label = pred.get("class", "")
        confidence = pred.get("confidence", 0)
        x = int(pred.get("x", 0))
        y = int(pred.get("y", 0))
        w = int(pred.get("width", 0))
        h = int(pred.get("height", 0))

        x1 = x - w // 2
        y1 = y - h // 2
        x2 = x + w // 2
        y2 = y + h // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {confidence:.2f}"
        cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return frame


def get_action(predictions):
    classes = {pred.get("class", "") for pred in predictions}
    if classes & STOP_CLASSES:
        return "STOP"
    if classes & SLOW_CLASSES:
        return "SLOW_DOWN"
    return "NORMAL"


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        raise SystemExit(1)

    frame_count = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to read frame.")
            break

        if frame_count % 3 == 0:
            predictions = run_detection(frame)

        frame = draw_detections(frame, predictions)
        action = get_action(predictions)
        print(f"Frame {frame_count} | Action: {action}")

        cv2.imshow("BFMC Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
