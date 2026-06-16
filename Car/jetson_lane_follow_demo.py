"""
BFMC Lane Following Test

This script runs on the Jetson and controls the BFMC car using USB camera-based
lane detection. It detects the white lane line on the black track and
adjusts the steering so the car stays centered while moving forward.

The program calculates the lane position and heading, then sends steering
and speed commands to the Brain running on the Raspberry Pi.

Before running:
1. Make sure the Jetson and Raspberry Pi are connected to the "BFMCDemocar" WiFi.
2. SSH into the Raspberry Pi and make sure main.py is running.
3. Run this script from the Jetson.

This is used to demonstrate basic autonomous lane following for the BFMC car.
Press ESC to stop the program.
"""


#!/usr/bin/env python3
import cv2
import socketio
import time
import numpy as np

PI_IP = "192.168.50.1"
SERVER = f"http://{PI_IP}:5005"
CAM_INDEX = 0   # change if needed

# -----------------------------
# Tuning
# -----------------------------
BASE_SPEED = 70
MIN_SPEED = 55

MAX_STEER = 140
KP_CTE = 110        # position error gain
KP_HEAD = 90        # heading error gain
STEER_SMOOTH = 0.65
LOST_LIMIT = 8

ROI_Y_START = 0.55
ROI_Y_END   = 0.95

LOWER_WHITE = np.array([0, 0, 190], dtype=np.uint8)
UPPER_WHITE = np.array([180, 65, 255], dtype=np.uint8)

TARGET_X_BIAS = 0

NEAR_ROW_FRAC = 0.78   # lower row in ROI
FAR_ROW_FRAC  = 0.38   # upper row in ROI
FAR2_ROW_FRAC = 0.25   # second far row (higher up, earlier curve detection)

MIN_ROW_PIXELS = 12


class BFMCController:
    def __init__(self, server):
        self.sio = socketio.Client()
        self.server = server

    def connect(self):
        print(f"Connecting to {self.server}")
        self.sio.connect(self.server)
        time.sleep(1.0)
        self.send('{"Name":"SessionAccess"}')
        time.sleep(1.0)
        self.send('{"Name":"Klem","Value":"30"}')
        time.sleep(0.5)
        self.send('{"Name":"DrivingMode","Value":"manual"}')
        time.sleep(0.5)

    def send(self, msg):
        self.sio.emit("message", msg)

    def set_speed(self, v):
        self.send(f'{{"Name":"SpeedMotor","Value":"{int(v)}"}}')

    def set_steer(self, a):
        self.send(f'{{"Name":"SteerMotor","Value":"{int(a)}"}}')

    def stop(self):
        self.set_speed(0)
        self.set_steer(0)

    def disconnect(self):
        try:
            self.stop()
            time.sleep(0.3)
            self.send('{"Name":"SessionEnd"}')
            time.sleep(0.3)
            self.sio.disconnect()
        except Exception:
            pass


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def row_center(mask, y):
    row = mask[y, :]
    xs = np.where(row > 0)[0]
    if len(xs) < MIN_ROW_PIXELS:
        return None
    return int(np.mean(xs))


def main():
    car = BFMCController(SERVER)
    cap = None

    last_steer = 0
    lost_count = 0

    try:
        car.connect()

        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            print("ERROR: camera failed to open")
            return

        print("Improved lane follower started. Press ESC to quit.")

        while True:
            ok, frame = cap.read()
            if not ok:
                car.stop()
                continue

            h, w = frame.shape[:2]
            y1 = int(ROI_Y_START * h)
            y2 = int(ROI_Y_END * h)
            roi = frame[y1:y2, :]

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, LOWER_WHITE, UPPER_WHITE)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            roi_h, roi_w = mask.shape[:2]
            near_y = int(NEAR_ROW_FRAC * (roi_h - 1))
            far_y = int(FAR_ROW_FRAC * (roi_h - 1))
            far2_y = int(FAR2_ROW_FRAC * (roi_h - 1))

            near_x = row_center(mask, near_y)
            far_x = row_center(mask, far_y)
            far2_x = row_center(mask, far2_y)

            dbg = frame.copy()
            cv2.rectangle(dbg, (0, y1), (w - 1, y2), (0, 255, 255), 2)

            target_x = (w // 2) + TARGET_X_BIAS
            cv2.line(dbg, (target_x, y1), (target_x, y2), (255, 255, 0), 2)

            if near_x is None:
                lost_count += 1
                cv2.putText(dbg, f"LINE LOST {lost_count}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                if lost_count < 4:
                    car.set_steer(last_steer)
                    car.set_speed(MIN_SPEED)   # creep, don't barrel ahead blind
                elif lost_count < 8:
                    car.set_steer(int(last_steer * 0.5))
                    car.set_speed(MIN_SPEED)
                else:
                    car.stop()

                cv2.imshow("lane_debug", dbg)
                cv2.imshow("lane_mask", mask)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            lost_count = 0

            # average far point with second far point if both valid; fallback to near
            if far_x is not None and far2_x is not None:
                far_x = (far_x + far2_x) // 2
            elif far_x is None and far2_x is not None:
                far_x = far2_x
            elif far_x is None:
                far_x = near_x

            # draw sampled points
            cv2.circle(dbg, (near_x, y1 + near_y), 7, (0, 255, 0), -1)
            cv2.circle(dbg, (far_x, y1 + far_y), 7, (0, 0, 255), -1)
            if far2_x is not None:
                cv2.circle(dbg, (far2_x, y1 + far2_y), 6, (255, 0, 255), -1)
            cv2.line(dbg, (near_x, y1 + near_y), (far_x, y1 + far_y), (0, 255, 0), 2)

            # cross-track error: near point vs target
            cte = (near_x - target_x) / (w / 2.0)

            # heading-like error: far-near
            heading = (far_x - near_x) / (w / 2.0)

            raw_steer = int((KP_CTE * cte) + (KP_HEAD * heading))
            raw_steer = clamp(raw_steer, -MAX_STEER, MAX_STEER)

            steer_cmd = int(STEER_SMOOTH * last_steer + (1.0 - STEER_SMOOTH) * raw_steer)
            last_steer = steer_cmd

            # slow down on bigger steering (three tiers)
            if abs(steer_cmd) > 100:
                speed_cmd = MIN_SPEED
            elif abs(steer_cmd) > 70:
                speed_cmd = 62
            elif abs(steer_cmd) > 40:
                speed_cmd = 72
            else:
                speed_cmd = BASE_SPEED

            car.set_steer(steer_cmd)
            car.set_speed(speed_cmd)

            cv2.putText(dbg, f"near_x={near_x} far_x={far_x} target={target_x}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(dbg, f"cte={cte:+.2f} head={heading:+.2f}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(dbg, f"steer={steer_cmd} speed={speed_cmd}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("lane_debug", dbg)
            cv2.imshow("lane_mask", mask)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            time.sleep(0.05)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        car.disconnect()


if __name__ == "__main__":
    main()
