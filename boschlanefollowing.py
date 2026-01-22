#!/usr/bin/env python3
import json
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

CAM_TOPIC = "/automobile/camera1/image_raw"
CMD_TOPIC = "/automobile/command"

class LaneFollower:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub = rospy.Publisher(CMD_TOPIC, String, queue_size=10)
        self.sub = rospy.Subscriber(CAM_TOPIC, Image, self.cb, queue_size=1)

        self.speed = rospy.get_param("~speed", 0.25)
        self.kp = rospy.get_param("~kp", 0.9)
        self.max_steer = rospy.get_param("~max_steer", 0.30)
        self.hz = rospy.get_param("~hz", 10)

        self.frame = None
        rospy.Timer(rospy.Duration(1.0 / self.hz), self.loop)

    def cb(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"cv_bridge: {e}")

def pub_speed(self, v):
    inner = {"action": "1", "speed": float(v)}
    outer = {"data": json.dumps(inner)}
    self.pub_cmd.publish(String(data=json.dumps(outer)))

def pub_steer(self, a):
    inner = {"action": "2", "steerAngle": float(a)}
    outer = {"data": json.dumps(inner)}
    self.pub_cmd.publish(String(data=json.dumps(outer)))

    def lane_error(self, bgr):
        h, w, _ = bgr.shape
        roi = bgr[int(0.55*h):h, 0:w]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, 180), (180, 70, 255))
        yellow = cv2.inRange(hsv, (15, 70, 70), (40, 255, 255))
        mask = cv2.bitwise_or(white, yellow)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(mask, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40,
                                minLineLength=40, maxLineGap=60)
        if lines is None:
            return None

        left_pts, right_pts = [], []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            slope = dy / float(dx)
            if abs(slope) < 0.3:
                continue
            if slope < 0:
                left_pts += [(x1, y1), (x2, y2)]
            else:
                right_pts += [(x1, y1), (x2, y2)]

        roi_h, roi_w = edges.shape
        y_eval = roi_h - 1

        def fit(pts):
            if len(pts) < 2:
                return None
            xs = np.array([p[0] for p in pts], dtype=np.float32)
            ys = np.array([p[1] for p in pts], dtype=np.float32)
            m, b = np.polyfit(xs, ys, 1)
            return m, b

        def x_at_y(mb, y):
            m, b = mb
            if m == 0:
                return None
            return (y - b) / m

        lf = fit(left_pts) if len(left_pts) >= 2 else None
        rf = fit(right_pts) if len(right_pts) >= 2 else None

        x_left = x_at_y(lf, y_eval) if lf else None
        x_right = x_at_y(rf, y_eval) if rf else None

        if x_left is not None and x_right is not None:
            lane_center = 0.5 * (x_left + x_right)
        elif x_left is not None:
            lane_center = x_left + 0.35 * roi_w
        elif x_right is not None:
            lane_center = x_right - 0.35 * roi_w
        else:
            return None

        img_center = 0.5 * roi_w
        err = (lane_center - img_center) / img_center  # -1..+1
        return float(err)

    def loop(self, _evt):
        if self.frame is None:
            return

        err = self.lane_error(self.frame)

        if err is None:
            self.pub_speed(0.0)
            self.pub_steer(0.0)
            return

        # Your sim: +steerAngle = left, -steerAngle = right
        steer = -self.kp * err
        steer = max(-self.max_steer, min(self.max_steer, steer))

        self.pub_steer(steer)
        self.pub_speed(self.speed)

def main():
    rospy.init_node("lane_follow_bfmc")
    LaneFollower()
    rospy.spin()

if __name__ == "__main__":
    main()
