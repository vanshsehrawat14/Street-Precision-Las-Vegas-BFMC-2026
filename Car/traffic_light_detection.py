#!/usr/bin/env python3
import cv2
import socketio
import time
import numpy as np

"""
BFMC Simple Traffic Light Test

This script runs on the Jetson and connects to the BFMC Brain on the Raspberry Pi.
It uses the camera to detect red and green traffic light colors.

Behavior:
- Red detected   -> stop
- Green detected -> drive forward
- No clear color -> keep last command briefly

Before running:
1. Make sure the Jetson and Raspberry Pi are connected to the "BFMCDemocar" WiFi.
2. SSH into the Raspberry Pi and make sure main.py is running.
3. Run this script from the Jetson.
"""

PI_IP = "192.168.50.1"
SERVER = f"http://{PI_IP}:5005"
CAM_INDEX = 0   # change if needed

# -----------------------------
# Tuning
# -----------------------------
DRIVE_SPEED = 90
MIN_AREA = 250          # minimum contour area to count as detection
HOLD_TIME = 1.0         # keep last detected state for this many seconds
SHOW_WINDOWS = True

sio = socketio.Client()

def send(msg):
    print("Sending:", msg)
    sio.emit("message", msg)

def set_speed(v):
    send(f'{{"Name":"SpeedMotor","Value":"{int(v)}"}}')

def set_steer(a):
    send(f'{{"Name":"SteerMotor","Value":"{int(a)}"}}')

def stop_car():
    set_speed(0)
    set_steer(0)

def largest_area(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area
            best_contour = c

    return max_area, best_contour

print("Connecting to Pi Brain...")
sio.connect(SERVER)
time.sleep(1)

send('{"Name":"SessionAccess"}')
time.sleep(1)
send('{"Name":"Klem","Value":"30"}')
send('{"Name":"DrivingMode","Value":"manual"}')

cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("ERROR: camera failed")
    send('{"Name":"SessionEnd"}')
    sio.disconnect()
    raise SystemExit

print("Starting traffic light detection... Press ESC to quit.")

last_state = "STOP"   # default to stopped until a clear signal shows up
last_seen_time = time.time()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            stop_car()
            continue

        h, w = frame.shape[:2]

        # focus on upper-middle area where traffic light is more likely to appear
        roi = frame[int(h * 0.10):int(h * 0.55), int(w * 0.25):int(w * 0.75)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # red has two HSV ranges
        lower_red1 = np.array([0, 120, 80], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 120, 80], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        lower_green = np.array([35, 80, 80], dtype=np.uint8)
        upper_green = np.array([90, 255, 255], dtype=np.uint8)

        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        red_area, red_contour = largest_area(red_mask)
        green_area, green_contour = largest_area(green_mask)

        detected_state = None

        if red_area > MIN_AREA and red_area > green_area:
            detected_state = "RED"
            last_state = "STOP"
            last_seen_time = time.time()

        elif green_area > MIN_AREA and green_area > red_area:
            detected_state = "GREEN"
            last_state = "GO"
            last_seen_time = time.time()

        # if nothing clear detected, keep last state for a short time
        if detected_state is None:
            if time.time() - last_seen_time > HOLD_TIME:
                last_state = "STOP"   # safer fallback

        if last_state == "STOP":
            print("RED / NO SIGNAL -> STOP")
            stop_car()
        else:
            print("GREEN -> GO")
            set_steer(0)
            set_speed(DRIVE_SPEED)

        dbg = frame.copy()

        # draw ROI box on full frame
        x1, y1 = int(w * 0.25), int(h * 0.10)
        x2, y2 = int(w * 0.75), int(h * 0.55)
        cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # draw contours back in ROI area
        roi_dbg = dbg[y1:y2, x1:x2]

        if red_contour is not None and red_area > MIN_AREA:
            x, y, ww, hh = cv2.boundingRect(red_contour)
            cv2.rectangle(roi_dbg, (x, y), (x + ww, y + hh), (0, 0, 255), 2)
            cv2.putText(roi_dbg, f"RED {int(red_area)}", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if green_contour is not None and green_area > MIN_AREA:
            x, y, ww, hh = cv2.boundingRect(green_contour)
            cv2.rectangle(roi_dbg, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.putText(roi_dbg, f"GREEN {int(green_area)}", (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(dbg, f"STATE: {last_state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if SHOW_WINDOWS:
            cv2.imshow("traffic_light_debug", dbg)
            cv2.imshow("red_mask", red_mask)
            cv2.imshow("green_mask", green_mask)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.05)

finally:
    cap.release()
    stop_car()
    send('{"Name":"SessionEnd"}')
    time.sleep(0.5)
    sio.disconnect()
    cv2.destroyAllWindows()
