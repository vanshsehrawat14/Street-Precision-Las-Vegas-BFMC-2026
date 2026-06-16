import os
import cv2
import base64
import requests
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
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


class ObjectDetectionNode:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_action = rospy.Publisher("/detection/action", String, queue_size=10)
        self.pub_debug = rospy.Publisher("/detection/debug_image", Image, queue_size=1)
        self.sub = rospy.Subscriber(
            "/automobile/camera1/image_raw", Image, self.cb, queue_size=1
        )
        self._frame_count = 0
        self._predictions = []
        rospy.loginfo("[object_detection] node ready")

    def cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"cv_bridge: {e}")
            return

        if self._frame_count % 3 == 0:
            self._predictions = run_detection(frame)

        self._frame_count += 1

        action = get_action(self._predictions)
        self.pub_action.publish(String(data=action))

        annotated = draw_detections(frame.copy(), self._predictions)
        try:
            self.pub_debug.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"debug publish: {e}")


if __name__ == "__main__":
    rospy.init_node("object_detection")
    ObjectDetectionNode()
    rospy.spin()
