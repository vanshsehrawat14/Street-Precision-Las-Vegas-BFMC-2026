"""
BFMC Simple Object Detection Test

This script runs on the Jetson and connects to the BFMC Brain on the Raspberry Pi.
It uses the USB camera to detect large objects in front of the car. If an object is
detected, the car stops. If the path is clear, the car drives forward.

Before running:
1. Make sure the Jetson and Raspberry Pi are connected to the "BFMCDemocar" WiFi.
2. SSH into the Raspberry Pi and make sure main.py is running.
3. Run this script from the Jetson.

This is mainly used to demonstrate basic vision control where the car stops when
something appears in front of it.
"""

import cv2
import socketio
import time

PI_IP = "192.168.50.1"
SERVER = f"http://{PI_IP}:5005"

CAM_INDEX = 0   # change if cam_test showed a different index

sio = socketio.Client()

def send(msg):
    print("Sending:", msg)
    sio.emit("message", msg)

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
    exit()

print("Starting vision control")

while True:

    ok, frame = cap.read()

    if not ok:
        continue

    h, w = frame.shape[:2]

    # center region of the frame
    roi = frame[int(h*0.4):int(h*0.9), int(w*0.3):int(w*0.7)]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),0)

    _,th = cv2.threshold(blur,80,255,cv2.THRESH_BINARY_INV)

    contours,_ = cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area > max_area:
            max_area = area

    if max_area > 18000:
        print("OBJECT DETECTED -> STOP")
        send('{"Name":"SpeedMotor","Value":"0"}')
    else:
        print("CLEAR -> DRIVE")
        send('{"Name":"SpeedMotor","Value":"100"}')
        send('{"Name":"SteerMotor","Value":"0"}')

    cv2.imshow("camera", frame)
    cv2.imshow("threshold", th)

    if cv2.waitKey(1) == 27:
        break

    time.sleep(0.05)

cap.release()

send('{"Name":"SpeedMotor","Value":"0"}')
send('{"Name":"SessionEnd"}')

sio.disconnect()

cv2.destroyAllWindows()
