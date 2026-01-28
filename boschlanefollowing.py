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
DEBUG_TOPIC = "/lane_follow/debug_image"


class LaneFollowerBFMC:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher(CMD_TOPIC, String, queue_size=10)
        self.pub_dbg = rospy.Publisher(DEBUG_TOPIC, Image, queue_size=1)
        self.sub = rospy.Subscriber(CAM_TOPIC, Image, self.cb, queue_size=1)

        # Controls (strong defaults for curves)
        self.hz = rospy.get_param("~hz", 20)
        self.v_max = rospy.get_param("~v_max", 0.16)
        self.v_min = rospy.get_param("~v_min", 0.08)

        self.max_steer = rospy.get_param("~max_steer", 0.95)
        self.min_steer = rospy.get_param("~min_steer", 0.18)

        self.k_stanley = rospy.get_param("~k_stanley", 5.0)
        self.k_heading = rospy.get_param("~k_heading", 2.6)

        self.steer_smooth = rospy.get_param("~steer_smooth", 0.25)
        self.deadband = rospy.get_param("~deadband", 0.01)

        self.invert_steer = rospy.get_param("~invert_steer", False)
        self.steer_key = rospy.get_param("~steer_key", "steerAngle")  

        # Vision (white-only) 
        self.roi_start = rospy.get_param("~roi_start", 0.55) # start ROI higher to "see" curve earlier
        self.thresh = rospy.get_param("~thresh", 200) # grayscale threshold for white
        self.morph_k = rospy.get_param("~morph_k", 5)

        # sliding window params
        self.nwindows = rospy.get_param("~nwindows", 9)
        self.margin = rospy.get_param("~margin", 90)
        self.minpix  = rospy.get_param("~minpix", 60)

        # lookahead for control (0..1 of ROI height; smaller = higher up)
        self.look_y       = rospy.get_param("~look_y", 0.45) # look further ahead for curves
        self.near_y       = rospy.get_param("~near_y", 0.85)

        # lane lost behavior
        self.lost_limit   = rospy.get_param("~lost_limit", 12)
        self.stop_on_lost = rospy.get_param("~stop_on_lost", True)

        self.frame = None
        self.last_steer = 0.0
        self.lost_count = 0

        rospy.Timer(rospy.Duration(1.0 / self.hz), self.loop)

        rospy.loginfo(f"[lane_follow_bfmc] ready | steer_key={self.steer_key}")

    def cb(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"cv_bridge: {e}")

    def pub_speed(self, v):
        self.pub_cmd.publish(String(data=json.dumps({"action": "1", "speed": float(v)})))

    def pub_steer(self, a):
        self.pub_cmd.publish(String(data=json.dumps({"action": "2", self.steer_key: float(a)})))

    @staticmethod
    def clamp(x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def make_debug_full(self, full_bgr, y0, roi_dbg):
        dbg = full_bgr.copy()
        h, w = dbg.shape[:2]
        roi_resized = cv2.resize(roi_dbg, (w, h - y0), interpolation=cv2.INTER_NEAREST)
        dbg[y0:h, 0:w] = roi_resized
        return dbg

    def sliding_window_fit(self, binary):
        """
        binary: 0/255 uint8 image (ROI)
        Returns: left_fit (a,b,c) for x = a*y^2 + b*y + c, right_fit, and debug overlay
        """
        h, w = binary.shape[:2]
        out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        # histogram of bottom half
        hist = np.sum(binary[h//2:, :] > 0, axis=0)

        midpoint = w // 2
        leftx_base = int(np.argmax(hist[:midpoint]))
        rightx_base = int(np.argmax(hist[midpoint:]) + midpoint)

        # Identify all nonzero pixels
        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # current positions
        leftx_current = leftx_base
        rightx_current = rightx_base

        window_height = h // self.nwindows
        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low = h - (window + 1) * window_height
            win_y_high = h - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # draw windows
            cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left)
            right_lane_inds.append(good_right)

            if len(good_left) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left]))
            if len(good_right) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds = np.concatenate(left_lane_inds) if len(left_lane_inds) else np.array([])
        right_lane_inds = np.concatenate(right_lane_inds) if len(right_lane_inds) else np.array([])

        left_fit = None
        right_fit = None

        if left_lane_inds.size > 200:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            left_fit = np.polyfit(lefty, leftx, 2) # x = a*y^2 + b*y + c
            out[lefty, leftx] = (255, 0, 0) # left pixels blue

        if right_lane_inds.size > 200:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            right_fit = np.polyfit(righty, rightx, 2)
            out[righty, rightx] = (0, 0, 255) # right pixels red

        return left_fit, right_fit, out

    def estimate_center_and_heading(self, bgr):
        h, w = bgr.shape[:2]
        y0 = int(self.roi_start * h)
        roi = bgr[y0:h, :, :]
        roi_h, roi_w = roi.shape[:2]
        cx = roi_w * 0.5

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY)

        k = max(3, int(self.morph_k) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        left_fit, right_fit, overlay = self.sliding_window_fit(binary)

        # choose y samples
        y_far = int(self.clamp(self.look_y, 0.0, 1.0) * (roi_h - 1))
        y_near = int(self.clamp(self.near_y, 0.0, 1.0) * (roi_h - 1))

        def x_from_fit(fit, y):
            if fit is None:
                return None
            return float(fit[0]*y*y + fit[1]*y + fit[2])

        xl_far = x_from_fit(left_fit, y_far)
        xr_far = x_from_fit(right_fit, y_far)
        xl_near = x_from_fit(left_fit, y_near)
        xr_near = x_from_fit(right_fit, y_near)

        # Need both lanes for true center (best). Fallback if only one exists.
        if xl_far is not None and xr_far is not None:
            c_far = 0.5*(xl_far + xr_far)
        elif xl_far is not None:
            c_far = xl_far + 0.42*roi_w
        elif xr_far is not None:
            c_far = xr_far - 0.42*roi_w
        else:
            dbg_full = self.make_debug_full(bgr, y0, overlay)
            cv2.putText(dbg_full, "NO LANES", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            return None, None, dbg_full

        if xl_near is not None and xr_near is not None:
            c_near = 0.5*(xl_near + xr_near)
        elif xl_near is not None:
            c_near = xl_near + 0.42*roi_w
        elif xr_near is not None:
            c_near = xr_near - 0.42*roi_w
        else:
            return None, None, self.make_debug_full(bgr, y0, overlay)

        # Cross-track error (normalized)
        cte = (c_far - cx) / cx
        cte = float(self.clamp(cte, -1.0, 1.0))

        # Heading error from centerline segment
        dx = (c_far - c_near)
        dy = (y_far - y_near) # typically negative
        heading_err = float(np.arctan2(dx, -dy + 1e-6))

        # draw fits + centers
        for yy in range(0, roi_h, 10):
            if left_fit is not None:
                xx = int(self.clamp(x_from_fit(left_fit, yy), 0, roi_w-1))
                cv2.circle(overlay, (xx, yy), 2, (255, 0, 0), -1)
            if right_fit is not None:
                xx = int(self.clamp(x_from_fit(right_fit, yy), 0, roi_w-1))
                cv2.circle(overlay, (xx, yy), 2, (0, 0, 255), -1)

        cv2.line(overlay, (int(cx), 0), (int(cx), roi_h-1), (0, 255, 255), 2)
        cv2.circle(overlay, (int(self.clamp(c_far, 0, roi_w-1)), y_far), 7, (0, 255, 0), -1)
        cv2.circle(overlay, (int(self.clamp(c_near, 0, roi_w-1)), y_near), 7, (0, 200, 0), -1)
        cv2.line(overlay, (int(c_near), y_near), (int(c_far), y_far), (0, 255, 0), 3)

        dbg_full = self.make_debug_full(bgr, y0, overlay)
        cv2.putText(dbg_full, f"cte={cte:+.3f} head={heading_err:+.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        return cte, heading_err, dbg_full

    def loop(self, _evt):
        if self.frame is None:
            return

        cte, head, dbg = self.estimate_center_and_heading(self.frame)

        # publish debug
        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))
        except:
            pass

        if cte is None:
            self.lost_count += 1
            if self.stop_on_lost and self.lost_count >= self.lost_limit:
                self.pub_speed(0.0)
            else:
                self.pub_speed(self.v_min)
            self.pub_steer(self.last_steer)
            return

        self.lost_count = 0

        if abs(cte) < self.deadband:
            cte = 0.0

        # Slow down a lot in turns so it can commit to the curve
        turn_mag = min(1.0, abs(head)/0.40 + abs(cte)/0.45)
        v = self.v_max - turn_mag*(self.v_max - self.v_min)
        v = float(self.clamp(v, self.v_min, self.v_max))

        # Stanley-like steering
        steer = (self.k_heading * head) + np.arctan2(self.k_stanley * cte, (v + 1e-3))

        if self.invert_steer:
            steer *= -1.0

        steer = float(self.clamp(steer, -self.max_steer, self.max_steer))

        # minimum kick so BFMC actually turns
        if abs(steer) > 1e-6 and abs(steer) < self.min_steer:
            steer = float(np.sign(steer) * self.min_steer)

        # smooth
        steer = self.steer_smooth*self.last_steer + (1.0-self.steer_smooth)*steer
        self.last_steer = steer

        self.pub_speed(v)
        self.pub_steer(steer)


def main():
    rospy.init_node("lane_follow_bfmc")
    LaneFollowerBFMC()
    rospy.spin()

if __name__ == "__main__":
    main()


