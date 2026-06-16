#!/usr/bin/env python3
import json
import numpy as np
import cv2
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import base64
import requests
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from fsm_detection import BFMCStateMachine, parse_predictions, update_fsm, draw_fsm_overlay
from src.semaphore_listener import SemaphoreListener

CAM_TOPIC   = "/automobile/camera1/image_raw"
CMD_TOPIC   = "/automobile/command"
DEBUG_TOPIC = "/lane_follow/debug_image"


class LaneFollowerBFMC:
    def __init__(self):
        self.bridge = CvBridge()
        self.pub_cmd = rospy.Publisher(CMD_TOPIC, String, queue_size=10)
        self.pub_dbg = rospy.Publisher(DEBUG_TOPIC, Image, queue_size=1)
        self.sub = rospy.Subscriber(CAM_TOPIC, Image, self.cb, queue_size=1)

        # -------- Control gains --------
        self.hz        = rospy.get_param("~hz", 20)
        self.v_max     = rospy.get_param("~v_max", 0.16)
        self.v_min     = rospy.get_param("~v_min", 0.08)

        self.max_steer = rospy.get_param("~max_steer", 10.0)
        self.min_steer = rospy.get_param("~min_steer", 7.22)

        self.k_stanley = rospy.get_param("~k_stanley", 7.5)
        self.k_heading = rospy.get_param("~k_heading", 3.8)

        self.steer_smooth = rospy.get_param("~steer_smooth", 0.20)
        self.deadband     = rospy.get_param("~deadband", 0.01)

        # Boost steering on sharp turns / when drifting (get back on track faster)
        self.turn_boost_gain       = rospy.get_param("~turn_boost_gain", 10.65)   # multiply steer when past threshold
        self.turn_boost_cte_thr   = rospy.get_param("~turn_boost_cte_thr", 0.07)  # |cte| above this -> boost (lower = boost sooner)
        self.turn_boost_heading_thr = rospy.get_param("~turn_boost_heading_thr", 0.15)  # |heading| above this -> boost

        self.invert_steer = rospy.get_param("~invert_steer", False)
        self.steer_key    = rospy.get_param("~steer_key", "steerAngle")  # try "steer" if needed

        # -------- Lane Vision (white-only lanes) --------
        self.roi_start    = rospy.get_param("~roi_start", 0.55)
        self.thresh       = rospy.get_param("~thresh", 200)
        self.morph_k      = rospy.get_param("~morph_k", 5)

        self.nwindows     = rospy.get_param("~nwindows", 9)
        self.margin       = rospy.get_param("~margin", 90)
        self.minpix       = rospy.get_param("~minpix", 60)

        # Minimum histogram column-sum to trust a lane base; prevents argmax
        # returning 0 on an empty half and spawning spurious left-edge windows.
        self.hist_min_peak = rospy.get_param("~hist_min_peak", 10)
        # Half-lane-width as a fraction of roi_w, used in the single-line fallback.
        # BFMC lane occupies ~60% of frame width, so half = 0.30.
        # Using 0.38 placed the estimated centre ~50px too far LEFT on right-only
        # detection (c_far = xr - 0.38*w vs correct xr - 0.30*w), producing a
        # persistent negative CTE and left steer on every straight frame.
        self.lane_half_w   = rospy.get_param("~lane_half_w", 0.30)

        self.look_y       = rospy.get_param("~look_y", 0.45)
        self.near_y       = rospy.get_param("~near_y", 0.85)

        self.lost_limit   = rospy.get_param("~lost_limit", 12)
        self.stop_on_lost = rospy.get_param("~stop_on_lost", True)

        # -------- Obstacle detection (front zone; motion-based) --------
        self.obs_enable         = rospy.get_param("~obs_enable", True)
        self.obs_roi_left       = rospy.get_param("~obs_roi_left", 0.35)
        self.obs_roi_right      = rospy.get_param("~obs_roi_right", 0.65)
        self.obs_roi_top        = rospy.get_param("~obs_roi_top", 0.55)
        self.obs_roi_bot        = rospy.get_param("~obs_roi_bot", 0.92)

        self.obs_motion_thr     = rospy.get_param("~obs_motion_thr", 18)
        self.obs_min_area       = rospy.get_param("~obs_min_area", 2500)
        self.obs_confirm_frames = rospy.get_param("~obs_confirm_frames", 2)

        # Emergency stop ONLY if very big (close)
        self.obs_emergency_area = rospy.get_param("~obs_emergency_area", 20000)
        # Stop when obstacle is in center and this big (too close to turn away)
        self.obs_stop_area      = rospy.get_param("~obs_stop_area", 9000)   # stop if center + area >= this
        self.obs_center_margin  = rospy.get_param("~obs_center_margin", 0.18)  # fraction of ROI width
        # Resume driving after obstacle cleared for this many consecutive frames
        self.obs_resume_clear_frames = rospy.get_param("~obs_resume_clear_frames", 5)

        # Throttle obstacle detection (set to 1 for fastest reaction)
        self.obs_every_n_frames = rospy.get_param("~obs_every_n_frames", 2)

        # -------- Avoidance (lane offset + steering push) --------
        self.avoid_enable        = rospy.get_param("~avoid_enable", True)
        self.avoid_offset_norm   = rospy.get_param("~avoid_offset_norm", 0.45)  # 0..1 of half-lane width
        self.avoid_smooth        = rospy.get_param("~avoid_smooth", 0.65)       # lower = quicker response
        self.avoid_slowdown      = rospy.get_param("~avoid_slowdown", True)
        self.avoid_v_scale       = rospy.get_param("~avoid_v_scale", 0.75)

        # NEW: direct steering push so it actually commits to moving sideways
        self.avoid_steer_bias    = rospy.get_param("~avoid_steer_bias", 0.55)

        # NEW: keep avoidance active for at least this long after detection
        self.avoid_min_time_sec  = rospy.get_param("~avoid_min_time_sec", 0.8)

        # Internal counters/state
        self._frame_count       = 0
        self._prev_obs_gray     = None
        self._obs_count         = 0
        self.obs_last_status    = "OBS: init"
        self._last_obs_side     = None
        self._last_obs_area     = 0.0

        self._avoid_dir         = 0.0   # -1 shift left, +1 shift right
        self._avoid_offset_filt = 0.0   # filtered normalized offset (-1..+1)
        self._avoid_until       = rospy.Time(0)

        # -------- Internal state --------
        self.frame = None
        self.last_steer = 0.0
        self.lost_count = 0

        # -------- Debug diagnostics (cached from previous frame, logged at loop top) --------
        self._dbg_left_fit    = False   # left_fit found?
        self._dbg_right_fit   = False   # right_fit found?
        self._dbg_leftx_base  = None    # histogram base x for left lane
        self._dbg_rightx_base = None    # histogram base x for right lane
        self._dbg_cte         = None    # cte from previous frame
        self._no_lane_count   = 0       # consecutive frames with no lanes at all

        self.state = "DRIVE"  # DRIVE or HALT
        self.halt_reason = ""
        self._halt_clear_count = 0  # consecutive frames with no stop-condition obstacle

        self._yolo_api_key = os.environ.get("ROBOFLOW_API_KEY", "")
        self._yolo_url = "https://detect.roboflow.com/bfmc-6btkg/3"
        self._yolo_every_n = 5
        self._yolo_frame_count = 0
        self._yolo_action = "NORMAL"
        self._yolo_speed_scale = 1.0
        self._last_predictions = []
        self.fsm = BFMCStateMachine()
        self._semaphore = SemaphoreListener()
        self._semaphore.start()

        rospy.Timer(rospy.Duration(1.0 / self.hz), self.loop)
        rospy.loginfo(
            f"[lane_follow_bfmc] ready | steer_key={self.steer_key} | obs_enable={self.obs_enable} | avoid_enable={self.avoid_enable}"
        )

    def cb(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn_throttle(2.0, f"cv_bridge: {e}")

    def pub_speed(self, v):
        rospy.loginfo(f"[CMD] speed={float(v):.4f}")
        self.pub_cmd.publish(String(data=json.dumps({"action": "1", "speed": float(v)})))

    def pub_steer(self, a):
        rospy.loginfo(f"[CMD] steer={float(a):.4f}  key={self.steer_key}")
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
        h, w = binary.shape[:2]
        out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        hist = np.sum(binary[h//2:, :] > 0, axis=0)
        midpoint = w // 2

        # Guard: np.argmax returns 0 on an all-zero array, which plants every
        # left window at x=0 and causes noise near the left edge to accumulate
        # into a spurious left_fit, pulling the computed centre hard left.
        # Only use a side's histogram peak if it has real pixels in it.
        left_peak  = int(np.max(hist[:midpoint])) if midpoint > 0 else 0
        right_peak = int(np.max(hist[midpoint:])) if midpoint < w  else 0

        leftx_base  = int(np.argmax(hist[:midpoint]))          if left_peak  >= self.hist_min_peak else None
        rightx_base = int(np.argmax(hist[midpoint:]) + midpoint) if right_peak >= self.hist_min_peak else None

        # Cache for diagnostic logging
        self._dbg_leftx_base  = leftx_base
        self._dbg_rightx_base = rightx_base

        rospy.loginfo_throttle(0.5,
            f"[HIST] lpeak={left_peak} lbase={leftx_base}  rpeak={right_peak} rbase={rightx_base}"
        )

        nonzero = binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current  = leftx_base
        rightx_current = rightx_base

        window_height = h // self.nwindows
        left_lane_inds  = []
        right_lane_inds = []

        for window in range(self.nwindows):
            win_y_low  = h - (window + 1) * window_height
            win_y_high = h -  window      * window_height

            if leftx_current is not None:
                win_xleft_low  = leftx_current - self.margin
                win_xleft_high = leftx_current + self.margin
                cv2.rectangle(out, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
                good_left = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                left_lane_inds.append(good_left)
                if len(good_left) > self.minpix:
                    leftx_current = int(np.mean(nonzerox[good_left]))

            if rightx_current is not None:
                win_xright_low  = rightx_current - self.margin
                win_xright_high = rightx_current + self.margin
                cv2.rectangle(out, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
                good_right = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                right_lane_inds.append(good_right)
                if len(good_right) > self.minpix:
                    rightx_current = int(np.mean(nonzerox[good_right]))

        left_lane_inds  = np.concatenate(left_lane_inds)  if left_lane_inds  else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        left_fit  = None
        right_fit = None

        if left_lane_inds.size > 200:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            left_fit = np.polyfit(lefty, leftx, 2)
            out[lefty, leftx] = (255, 0, 0)

        if right_lane_inds.size > 200:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            right_fit = np.polyfit(righty, rightx, 2)
            out[righty, rightx] = (0, 0, 255)

        return left_fit, right_fit, out

    def estimate_center_and_heading(self, bgr, avoid_norm=0.0):
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
        self._dbg_left_fit  = left_fit  is not None
        self._dbg_right_fit = right_fit is not None

        y_far  = int(self.clamp(self.look_y, 0.0, 1.0) * (roi_h - 1))
        y_near = int(self.clamp(self.near_y, 0.0, 1.0) * (roi_h - 1))

        def x_from_fit(fit, y):
            if fit is None:
                return None
            return float(fit[0]*y*y + fit[1]*y + fit[2])

        xl_far  = x_from_fit(left_fit,  y_far)
        xr_far  = x_from_fit(right_fit, y_far)
        xl_near = x_from_fit(left_fit,  y_near)
        xr_near = x_from_fit(right_fit, y_near)

        if xl_far is not None and xr_far is not None:
            c_far = 0.5*(xl_far + xr_far)
        elif xl_far is not None:
            c_far = xl_far + self.lane_half_w*roi_w
        elif xr_far is not None:
            c_far = xr_far - self.lane_half_w*roi_w
        else:
            dbg_full = self.make_debug_full(bgr, y0, overlay)
            cv2.putText(dbg_full, "NO LANES", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            return None, None, dbg_full

        if xl_near is not None and xr_near is not None:
            c_near = 0.5*(xl_near + xr_near)
        elif xl_near is not None:
            c_near = xl_near + self.lane_half_w*roi_w
        elif xr_near is not None:
            c_near = xr_near - self.lane_half_w*roi_w
        else:
            return None, None, self.make_debug_full(bgr, y0, overlay)

        # Apply avoidance shift in pixels
        avoid_norm = float(self.clamp(avoid_norm, -1.0, 1.0))
        shift_px = avoid_norm * (roi_w * 0.5)
        c_far  = c_far + shift_px
        c_near = c_near + shift_px

        cte = (c_far - cx) / cx
        cte = float(self.clamp(cte, -1.0, 1.0))

        rospy.loginfo_throttle(0.5,
            f"[LANES] lbase={self._dbg_leftx_base} rbase={self._dbg_rightx_base} "
            f"center_px={c_far:.1f} frame_cx={cx:.1f} cte={cte:+.3f} steer={self.last_steer:+.2f}"
        )

        dx = (c_far - c_near)
        dy = (y_far - y_near)
        heading_err = float(np.arctan2(dx, -dy + 1e-6))

        # draw fits + centers
        for yy in range(0, roi_h, 10):
            if left_fit is not None:
                xx = int(self.clamp(x_from_fit(left_fit, yy), 0, roi_w-1))
                cv2.circle(overlay, (xx, yy), 2, (255, 0, 0), -1)
            if right_fit is not None:
                xx = int(self.clamp(x_from_fit(right_fit, yy), 0, roi_w-1))
                cv2.circle(overlay, (xx, yy), 2, (0, 0, 255), -1)

        # nominal center line
        cv2.line(overlay, (int(cx), 0), (int(cx), roi_h-1), (0, 255, 255), 2)

        # shifted target line
        tgt = int(self.clamp(cx + shift_px, 0, roi_w-1))
        cv2.line(overlay, (tgt, 0), (tgt, roi_h-1), (255, 255, 0), 2)

        cv2.circle(overlay, (int(self.clamp(c_far, 0, roi_w-1)), y_far), 7, (0, 255, 0), -1)
        cv2.circle(overlay, (int(self.clamp(c_near, 0, roi_w-1)), y_near), 7, (0, 200, 0), -1)
        cv2.line(overlay, (int(self.clamp(c_near, 0, roi_w-1)), y_near),
                 (int(self.clamp(c_far, 0, roi_w-1)), y_far), (0, 255, 0), 3)

        dbg_full = self.make_debug_full(bgr, y0, overlay)
        cv2.putText(dbg_full, f"cte={cte:+.3f} head={heading_err:+.3f} avoid={avoid_norm:+.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
        dbg_full = draw_fsm_overlay(dbg_full, self.fsm, self._yolo_action, self._yolo_speed_scale, self._last_predictions)

        return cte, heading_err, dbg_full

    def _yolo_detect(self, bgr):
        try:
            _, buf = cv2.imencode(".jpg", bgr)
            b64 = base64.b64encode(buf).decode("utf-8")
            r = requests.post(
                self._yolo_url,
                params={"api_key": self._yolo_api_key, "confidence": 45},
                data=b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=3,
            )
            raw = r.json().get("predictions", [])
            self._last_predictions = raw
            det = parse_predictions(raw, bgr)
            v2x_state = self._semaphore.get_nearest_state()
            if v2x_state is not None:
                v2x_color = SemaphoreListener.state_to_color(v2x_state)
                if v2x_color and not det.traffic_light_color:
                    det.traffic_light_color = v2x_color
                    det.classes.add(f"traffic_light_{v2x_color}")
                    rospy.loginfo_throttle(2.0, f"[V2X] Using semaphore state: {v2x_color}")
            now = rospy.Time.now().to_sec()
            action, speed_scale, self.fsm = update_fsm(self.fsm, det, now)
            self._yolo_speed_scale = speed_scale
            return action
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"[YOLO] {e}")
            return "NORMAL"

    def detect_obstacle_motion(self, bgr):
        """
        Returns (found, side, area, dbg_img)
        side: "left"|"right"|"center"|None
        """
        h, w = bgr.shape[:2]
        x1 = int(self.clamp(self.obs_roi_left, 0.0, 1.0) * w)
        x2 = int(self.clamp(self.obs_roi_right, 0.0, 1.0) * w)
        y1 = int(self.clamp(self.obs_roi_top, 0.0, 1.0) * h)
        y2 = int(self.clamp(self.obs_roi_bot, 0.0, 1.0) * h)

        if x2 <= x1 + 10 or y2 <= y1 + 10:
            x1, x2 = int(0.35*w), int(0.65*w)
            y1, y2 = int(0.55*h), int(0.92*h)

        roi = bgr[y1:y2, x1:x2, :]
        roi_h, roi_w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        dbg = bgr.copy()
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)

        if self._prev_obs_gray is None or self._prev_obs_gray.shape != gray.shape:
            self._prev_obs_gray = gray
            self.obs_last_status = "OBS: init"
            cv2.putText(dbg, self.obs_last_status, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return False, None, 0.0, dbg

        diff = cv2.absdiff(gray, self._prev_obs_gray)
        self._prev_obs_gray = gray

        _, th = cv2.threshold(diff, int(self.obs_motion_thr), 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, iterations=2)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        biggest = 0.0
        best_bbox = None

        for c in cnts:
            area = float(cv2.contourArea(c))
            if area > biggest:
                biggest = area
                best_bbox = cv2.boundingRect(c)

        found = False
        side = None

        if best_bbox is not None and biggest >= float(self.obs_min_area):
            found = True
            bx, by, bww, bhh = best_bbox
            cv2.rectangle(dbg, (x1 + bx, y1 + by), (x1 + bx + bww, y1 + by + bhh), (0, 0, 255), 2)

            obj_cx = (bx + 0.5 * bww)
            roi_mid = 0.5 * roi_w
            m = float(self.obs_center_margin) * roi_w

            if obj_cx < roi_mid - m:
                side = "left"
            elif obj_cx > roi_mid + m:
                side = "right"
            else:
                side = "center"

            cv2.circle(dbg, (x1 + int(obj_cx), y1 + int(by + 0.5*bhh)), 6, (0, 0, 255), -1)

        self.obs_last_status = (
            f"OBS: {'FOUND' if found else 'none'} side={side} biggest={biggest:.0f} thr={int(self.obs_motion_thr)} minA={int(self.obs_min_area)}"
        )
        cv2.putText(dbg, self.obs_last_status, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return found, side, biggest, dbg

    def loop(self, _evt):
        if self.frame is None:
            return

        # -------- Per-frame diagnostic log (values from previous frame) --------
        _cte_s = f"{self._dbg_cte:+.3f}" if self._dbg_cte is not None else "  None"
        rospy.loginfo_throttle(0.5,
            f"[DIAG] "
            f"L={'YES' if self._dbg_left_fit  else 'NO ':3s} "
            f"R={'YES' if self._dbg_right_fit else 'NO ':3s} "
            f"lbase={str(self._dbg_leftx_base):>5s} "
            f"rbase={str(self._dbg_rightx_base):>5s} "
            f"cte={_cte_s} "
            f"steer={self.last_steer:+.2f} "
            f"FSM={self.fsm.state.name} "
            f"yolo={self._yolo_action}"
        )

        self._yolo_frame_count += 1
        if self._yolo_frame_count % self._yolo_every_n == 0:
            self._yolo_action = self._yolo_detect(self.frame)

        # -------- HALT state: stop but keep checking if obstacle is gone --------
        if self.state == "HALT":
            # Continuously check: if path is clear for N frames, resume driving
            if self.obs_enable:
                obs_found, side, area, obs_dbg = self.detect_obstacle_motion(self.frame)
                still_must_stop = (
                    obs_found
                    and (
                        area >= float(self.obs_emergency_area)
                        or (side == "center" and area >= float(self.obs_stop_area))
                    )
                )
                if still_must_stop:
                    self._halt_clear_count = 0
                else:
                    self._halt_clear_count += 1
                if self._halt_clear_count >= int(self.obs_resume_clear_frames):
                    self.state = "DRIVE"
                    self.halt_reason = ""
                    self._halt_clear_count = 0
                    rospy.loginfo("[lane_follow_bfmc] Obstacle cleared, resuming lane follow")
                    # fall through to normal loop below (don't return)
                else:
                    self.pub_speed(0.0)
                    self.pub_steer(self.last_steer)
                    try:
                        cv_dbg = self.frame.copy()
                        cv2.putText(cv_dbg, f"HALTED: {self.halt_reason}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                        cv2.putText(cv_dbg, self.obs_last_status, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(cv_dbg, f"Resume in {int(self.obs_resume_clear_frames) - self._halt_clear_count} clear frames",
                                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(cv_dbg, encoding="bgr8"))
                    except:
                        pass
                    return
            else:
                # obs disabled while halted: stay halted (no automatic resume)
                self.pub_speed(0.0)
                self.pub_steer(self.last_steer)
                try:
                    cv_dbg = self.frame.copy()
                    cv2.putText(cv_dbg, f"HALTED: {self.halt_reason}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                    self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(cv_dbg, encoding="bgr8"))
                except:
                    pass
                return

        # -------- Obstacle detect + Avoidance update --------
        obs_dbg_for_pub = None
        obs_found_now = False

        if self.obs_enable:
            self._frame_count += 1
            do_check = (self._frame_count % int(max(1, self.obs_every_n_frames))) == 0

            if do_check:
                obs_found, side, area, obs_dbg = self.detect_obstacle_motion(self.frame)
                obs_dbg_for_pub = obs_dbg
                obs_found_now = obs_found

                self._last_obs_side = side
                self._last_obs_area = area

                if obs_found:
                    self._obs_count += 1
                else:
                    self._obs_count = max(0, self._obs_count - 1)

                # Avoidance decision: shift AWAY from obstacle side
                if self.avoid_enable and obs_found:
                    if side == "left":
                        self._avoid_dir = +1.0
                    elif side == "right":
                        self._avoid_dir = -1.0
                    else:
                        # center obstacle: keep last direction if we had one, else default right
                        self._avoid_dir = self._avoid_dir if self._avoid_dir != 0.0 else +1.0

                    # keep avoidance active for at least this long
                    self._avoid_until = rospy.Time.now() + rospy.Duration(float(self.avoid_min_time_sec))

                # Emergency stop if very big (very close), any position
                if obs_found and (area >= float(self.obs_emergency_area)):
                    if self._obs_count >= int(self.obs_confirm_frames):
                        self.state = "HALT"
                        self.halt_reason = "Emergency obstacle (too close)"
                        self.pub_speed(0.0)
                        self.pub_steer(self.last_steer)
                        try:
                            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(obs_dbg, encoding="bgr8"))
                        except:
                            pass
                        return

                # Stop when too close and can't turn away (obstacle in center)
                if obs_found and side == "center" and (area >= float(self.obs_stop_area)):
                    if self._obs_count >= int(self.obs_confirm_frames):
                        self.state = "HALT"
                        self.halt_reason = "Obstacle too close (center, can't turn away)"
                        self.pub_speed(0.0)
                        self.pub_steer(self.last_steer)
                        try:
                            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(obs_dbg, encoding="bgr8"))
                        except:
                            pass
                        return

            # Clear avoid if time expired
            if rospy.Time.now() >= self._avoid_until:
                self._avoid_dir = 0.0

        # Compute normalized avoidance target
        target_avoid = 0.0
        if self.avoid_enable and self._avoid_dir != 0.0:
            target_avoid = float(self._avoid_dir) * float(self.avoid_offset_norm)

        # Smooth avoidance offset
        a = float(self.clamp(self.avoid_smooth, 0.0, 0.99))
        self._avoid_offset_filt = a * self._avoid_offset_filt + (1.0 - a) * target_avoid
        avoid_norm = float(self.clamp(self._avoid_offset_filt, -1.0, 1.0))

        # -------- Lane following (with avoid_norm bias) --------
        cte, head, dbg = self.estimate_center_and_heading(self.frame, avoid_norm=avoid_norm)
        self._dbg_cte = cte  # cache for next frame's top-of-loop log

        # Always use lane view for debug (avoids flicker from alternating lane/obstacle image)
        if dbg is not None:
            cv2.putText(dbg, f"avoid_norm={avoid_norm:+.2f} dir={self._avoid_dir:+.0f} until={(self._avoid_until - rospy.Time.now()).to_sec():+.2f}s",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
            cv2.putText(dbg, self.obs_last_status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        try:
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))
        except:
            pass

        if cte is None:
            self._no_lane_count += 1
            self.lost_count += 1
            if self._no_lane_count >= 3:
                rospy.logwarn_throttle(1.0,
                    f"[lane_follow_bfmc] NO LANES DETECTED - STOPPING "
                    f"({self._no_lane_count} consecutive frames)")
                self.pub_speed(0.0)
                self.pub_steer(0.0)
                self.last_steer = 0.0
            elif self.stop_on_lost and self.lost_count >= self.lost_limit:
                self.pub_speed(0.0)
                self.pub_steer(self.last_steer)
            else:
                self.pub_speed(self.v_min)
                self.pub_steer(self.last_steer)
            return

        self._no_lane_count = 0
        self.lost_count = 0

        if abs(cte) < self.deadband:
            cte = 0.0

        # YOLO action overrides
        if self._yolo_action == "STOP":
            self.pub_speed(0.0)
            self.pub_steer(self.last_steer)
            return

        # Speed selection (slow in turns, slower while avoiding)
        turn_mag = min(1.0, abs(head)/0.40 + abs(cte)/0.45)
        v = self.v_max - turn_mag*(self.v_max - self.v_min)
        v = v * self._yolo_speed_scale
        v = float(self.clamp(v, self.v_min, self.v_max))

        if self.avoid_slowdown and abs(avoid_norm) > 0.05:
            v = float(self.clamp(v * float(self.avoid_v_scale), self.v_min, self.v_max))

        # Steering (Stanley-like) + FORCE bias to commit sideways
        steer = (self.k_heading * head) + np.arctan2(self.k_stanley * cte, (v + 1e-3))
        steer += float(self.avoid_steer_bias) * float(avoid_norm)

        # On sharp turns or when drifting, boost steering to recover sooner
        if abs(cte) >= float(self.turn_boost_cte_thr) or abs(head) >= float(self.turn_boost_heading_thr):
            steer *= float(self.turn_boost_gain)

        if self.invert_steer:
            steer *= -1.0

        steer = float(self.clamp(steer, -self.max_steer, self.max_steer))

        if abs(steer) > 1e-6 and abs(steer) < self.min_steer:
            steer = float(np.sign(steer) * self.min_steer)

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


