# Simulation: ROS Pipeline

ROS 1 (Noetic) nodes for the BFMC Gazebo simulator. The pipeline reads the
car's camera, follows the lane, reacts to obstacles, and obeys traffic rules.

## Files

| File | Role |
| --- | --- |
| `lane_follow_bfmc.py` | Integrated driving node: lane following, obstacle detection and avoidance, YOLO and FSM traffic handling, V2X. |
| `lane_follow_baseline.py` | Pure lane-following baseline (sliding window and Stanley control). Reference for the integrated node. |
| `src/fsm_detection.py` | Finite-state machine and detection parsing. |
| `src/object_detection.py` | Standalone detection node publishing a driving action. |
| `src/semaphore_listener.py` | V2X semaphore receiver (UDP port 5007). |

## How it works

Every frame, the node estimates the lane center and heading from a thresholded
region of interest using a sliding-window polynomial fit, then computes a
Stanley steering command from cross-track and heading error. Speed is reduced in
turns and while avoiding obstacles.

Obstacle detection runs frame differencing over a front region of interest. A
detected obstacle shifts the target lane offset to steer around it, and a close,
centered obstacle triggers a halt that clears once the path is free.

Roboflow predictions are parsed into a `Detection` and fed to the FSM in
`src/fsm_detection.py`, which handles stop signs, traffic lights, and
pedestrians and returns a driving action and a speed scale. When the camera
cannot read a traffic light, the V2X listener supplies its state.

## Running

Requires a ROS 1 (Noetic) install and the BFMC simulator. ROS provides `rospy`
and `cv_bridge`; install the remaining Python dependencies from the repository
root with `pip install -r requirements.txt`.

```bash
# Detection uses the Roboflow inference API
export ROBOFLOW_API_KEY=your_api_key_here

rosrun <your_package> lane_follow_bfmc.py
```

Tuning is exposed through ROS parameters in the private `~` namespace: speed
limits, steering gains, region-of-interest bounds, and the obstacle and
avoidance thresholds. See the parameter definitions at the top of
`lane_follow_bfmc.py`.

The node publishes an annotated debug image on `/lane_follow/debug_image`.
