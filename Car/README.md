# Car Demos

Four scripts that run on the Jetson to test and demo the physical 1:10 BFMC car.
Each connects to the Brain (Raspberry Pi) over Wi-Fi and sends driving commands.

## Before running

1. Connect both the Jetson and the Pi to the `BFMCDemocar` Wi-Fi.
2. SSH into the Pi and confirm `main.py` is running.
3. Run the script from the Jetson, not the Pi.
4. The Pi's IP is hardcoded as `192.168.50.1`. Change it only if your network differs.
5. The USB camera defaults to index `0`. If it does not open, set `CAM_INDEX = 1` in the script.

## Scripts

### jetson_motor_test.py

Sanity check for motors and communication. The car drives forward, turns right,
turns left, then stops. No camera needed. Run this first when setting up: if it
works, the Jetson-to-Pi link and the motors are good.

```bash
python3 jetson_motor_test.py
```

The script runs to completion and exits on its own. Give the car room to move.

### jetson_lane_follow_demo.py

Autonomous lane following. The car uses the USB camera to find the white lane
line on the black track and steers to stay centered. It slows on sharper turns
and stops if it loses the line.

```bash
python3 jetson_lane_follow_demo.py
```

A debug window shows the camera view and the detected lane. Press ESC to stop;
the car stops and disconnects cleanly.

Tunable values at the top of the file:
- `BASE_SPEED`, `MIN_SPEED`: speed range.
- `KP_CTE`, `KP_HEAD`: position correction vs heading correction.
- `TARGET_X_BIAS`: shift the target left or right if the car runs off-center.

### jetson_stop_on_object.py

The car drives forward and stops when something large appears ahead. It
thresholds the camera feed with no model: if the blob in the center region
exceeds 18000 pixels, it stops.

```bash
python3 jetson_stop_on_object.py
```

Two windows open: the raw feed and the thresholded view. Press ESC to quit.
Detection covers the middle of the frame and ignores the edges. Adjust the
18000-pixel threshold in the script if needed.

### traffic_light_detection.py

Detects red and green traffic lights by color. Red stops, green drives. If it
loses a clear signal, it holds the last state for one second, then defaults to
stopped.

```bash
python3 traffic_light_detection.py
```

Three windows open: the camera view with a box around detections, and separate
red and green masks. Press ESC to stop.

Tunable values at the top of the file:
- `DRIVE_SPEED`: speed on green.
- `MIN_AREA`: minimum blob size to count as a detection (raise it to cut false positives).
- `HOLD_TIME`: how long the last signal is held before defaulting to stopped.
- `SHOW_WINDOWS`: set to `False` to hide the debug windows.

## Common issues

**`ERROR: camera failed to open`.** The USB camera is not on index 0. Try
`CAM_INDEX = 1` or `2`.

**Script connects but the car does not move.** Confirm `main.py` is running on
the Pi and the car is powered on.

**Steering is off (lane follow).** Check that the camera points at the track.
The lane follower expects the white line in the lower part of the frame.

**Traffic light not detected.** Keep the light in the upper-center of the frame,
where the ROI is. Lower `MIN_AREA` if the light looks small. Lighting matters.

**Connection error or timeout.** Confirm both devices are on `BFMCDemocar` Wi-Fi
and the Pi is at `192.168.50.1`. Run `ping 192.168.50.1` from the Jetson.
