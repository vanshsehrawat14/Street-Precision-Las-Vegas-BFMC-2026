"""
BFMC Manual Driving Test

This script connects the Jetson to the BFMC Brain (running on the Raspberry Pi)
and sends simple driving commands to test communication and motor control.
The car will drive forward, turn right, turn left, then stop.

Before running:
1. Make sure the Jetson and Raspberry Pi are connected to the "BFMCDemocar" WiFi.
2. SSH into the Raspberry Pi and make sure main.py is running.
3. On another monitor, run this script from the Jetson.

This is mainly used to quickly test that the Jetson ↔ Pi communication and
car movement are working.
"""

#!/usr/bin/env python3
import socketio
import time

sio = socketio.Client()

@sio.event
def connect():
    print("Connected")

@sio.event
def disconnect():
    print("Disconnected")

@sio.on("session_access")
def on_session_access(data):
    print("session_access:", data)

def send(msg):
    print("Sending:", msg)
    sio.emit("message", msg)
    time.sleep(0.3)

print("Connecting to Brain...")
sio.connect("http://192.168.50.1:5005")

time.sleep(1)

# Start session
send('{"Name": "SessionAccess"}')
send('{"Name": "Klem", "Value": "30"}')
send('{"Name": "DrivingMode", "Value": "manual"}')

print("\nDriving forward...")
send('{"Name": "SteerMotor", "Value": "0"}')
send('{"Name": "SpeedMotor", "Value": "120"}')
time.sleep(4)

print("\nTurning RIGHT...")
send('{"Name": "SteerMotor", "Value": "80"}')
time.sleep(4)

print("\nTurning LEFT...")
send('{"Name": "SteerMotor", "Value": "-80"}')
time.sleep(4)

print("\nStraight again...")
send('{"Name": "SteerMotor", "Value": "0"}')
time.sleep(3)

print("\nStopping...")
send('{"Name": "SpeedMotor", "Value": "0"}')
send('{"Name": "SteerMotor", "Value": "0"}')

send('{"Name": "SessionEnd"}')

time.sleep(1)
sio.disconnect()
