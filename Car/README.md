# Demo Codes

Four scripts you can run on the Jetson to test and demo the BFMC car. Each one connects to the Brain (Raspberry Pi) over WiFi and sends driving commands.

---

## Before Running Anything

1. Make sure the Jetson **and** the Pi are both connected to the `BFMCDemocar` WiFi.
2. SSH into the Pi and confirm `main.py` is already running.
3. Run the script from the Jetson, not the Pi.
4. The Pi's IP is hardcoded as `192.168.50.1`. Don't change it unless your network is different.
5. The USB camera defaults to index `0`. If the camera doesn't open, try changing `CAM_INDEX = 1` in the script.

---

## Scripts

### `jetson_motor_test.py`

**What it does:** Quick sanity check for motors and communication. The car drives forward, turns right, turns left, then stops. That's it. No camera needed.

**When to use it:** First thing you run when setting up. If this works, the Jetson ↔ Pi connection is good and the motors are responding.

**How to run:**
```bash
python3 jetson_motor_test.py
```

The script runs on its own and exits automatically. Just make sure the car has space to move.

---

### `jetson_lane_follow_demo.py`

**What it does:** Autonomous lane following. The car uses the USB camera to find the white lane line on the black track and steers itself to stay centered. It slows down on sharper turns and stops if it loses the line completely.

**How to run:**
```bash
python3 jetson_lane_follow_demo.py
```

A debug window pops up showing what the camera sees and where the script thinks the lane is. Press **ESC** to stop. The car will stop and disconnect cleanly.

**Things you might want to tweak (top of the file):**
- `BASE_SPEED` / `MIN_SPEED` — how fast the car goes
- `KP_CTE` / `KP_HEAD` — how aggressively it corrects position vs heading
- `TARGET_X_BIAS` — shift the target left/right if the car is consistently off-center

---

### `jetson_stop_on_object.py`

**What it does:** The car drives forward and stops when something large appears in front of it. It uses basic thresholding on the camera feed — no fancy model. If the detected blob in the center region is big enough, it stops.

**How to run:**
```bash
python3 jetson_stop_on_object.py
```

Two windows open: the raw camera feed and the thresholded view. Press **ESC** to quit.

**Note:** The detection area is the middle of the frame. It ignores edges. The threshold for "big enough to stop" is an area of 18000 pixels — you can adjust that in the script if it's too sensitive or not sensitive enough.

---

### `traffic_light_detection.py`

**What it does:** Detects red and green traffic lights using color detection. Red = stop, green = go. If it loses sight of a clear signal, it holds the last state for 1 second, then defaults to stopped.

**How to run:**
```bash
python3 traffic_light_detection.py
```

Three debug windows open: the main camera view with a bounding box around detections, and separate red/green masks. Press **ESC** to stop.

**Things you might want to tweak (top of the file):**
- `DRIVE_SPEED` — speed when the light is green
- `MIN_AREA` — minimum blob size to count as a detection (raise it if you're getting false positives)
- `HOLD_TIME` — how long it remembers the last signal before defaulting to stopped
- `SHOW_WINDOWS` — set to `False` if you don't want the debug windows

---

## Common Issues

**"ERROR: camera failed to open"**
The USB camera isn't on index 0. Try `CAM_INDEX = 1` or `CAM_INDEX = 2` in the script.

**Script connects but car doesn't move**
Make sure `main.py` is actually running on the Pi. SSH in and check. Also confirm the car is powered on (not just the Pi).

**Car moves but steering is way off (lane follow)**
Check that the camera is pointed at the track. The lane follower expects the white line to be visible in the lower portion of the frame. Tilt or reposition the camera if needed.

**Traffic light not detecting**
Make sure the light is in the upper-center of the frame — that's where the ROI is. Try lowering `MIN_AREA` if the light looks small in the frame. Lighting conditions matter a lot here.

**Connection error / timeout**
Double-check both devices are on `BFMCDemocar` WiFi and the Pi IP is `192.168.50.1`. Run a quick `ping 192.168.50.1` from the Jetson to verify.
