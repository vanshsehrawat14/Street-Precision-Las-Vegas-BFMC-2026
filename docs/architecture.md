# System Architecture

Team Street Precision Las Vegas, Bosch Future Mobility Challenge 2026.

The project has two software targets that share the same control ideas: a ROS
pipeline for the BFMC simulator, and Jetson scripts that drive the physical 1:10
car. Both produce two values each frame, a speed and a steering angle, from a
single forward camera.

## Hardware

| Layer | Component | Role |
| --- | --- | --- |
| Sensing | Camera | Forward-facing video for lane and object detection. The physical car uses the Raspberry Pi CSI camera; the Jetson demos use a USB camera. |
| High-level compute | Raspberry Pi 4 (8 GB), the "Brain" | Runs the BFMC Brain software: image processing, the web dashboard, and a Socket.IO command server on port `5005`. |
| Demo compute | NVIDIA Jetson | Runs the vision in `Car/` and sends driving commands to the Brain over Wi-Fi. |
| Low-level control | STM32 Nucleo | Receives serial commands from the Pi and drives the steering servo and DC motor. |
| Connectivity | Wi-Fi (SSH, TCP, Socket.IO) | Links the Jetson and a host PC to the car for control and monitoring. |

## Software subsystems

### Simulation: ROS pipeline (`Simulation/`)

ROS 1 nodes written for the BFMC Gazebo simulator.

- **`lane_follow_bfmc.py`** is the integrated driving node. It combines:
  - White-lane detection with a sliding-window polynomial fit, and Stanley
    steering from cross-track and heading error.
  - Motion-based obstacle detection in a front region of interest, with a
    lane-offset avoidance maneuver and an emergency stop when an obstacle is
    centered and close.
  - Roboflow YOLO detection feeding a finite-state machine that handles stop
    signs, traffic lights, and pedestrians/crosswalks.
  - A V2X semaphore listener that supplies traffic-light state when the camera
    cannot read it directly.
- **`lane_follow_baseline.py`** is the pure lane-following baseline (sliding
  window and Stanley control, city/highway speed targets, no detection). Kept as
  a clean reference point for the integrated node.
- **`src/fsm_detection.py`** holds the finite-state machine and detection parsing.
- **`src/object_detection.py`** is a standalone detection node that publishes a
  driving action, an alternative to the detection embedded in the integrated node.
- **`src/semaphore_listener.py`** is the V2X receiver (UDP port `5007`).

ROS topics:

| Topic | Direction | Purpose |
| --- | --- | --- |
| `/automobile/camera1/image_raw` | in | Camera feed |
| `/automobile/command` | out | Speed and steering commands (JSON) |
| `/lane_follow/debug_image` | out | Annotated debug view |
| `/detection/action`, `/detection/debug_image` | out | Standalone detection node |
| `/v2x/semaphore` | out | Decoded semaphore state |

Commands are JSON strings: `{"action": "1", "speed": v}` for speed and
`{"action": "2", "steerAngle": a}` for steering.

### Car: Jetson demos (`Car/`)

Scripts that run on the Jetson and drive the physical car by sending Socket.IO
`message` events to the Brain at `http://192.168.50.1:5005`. Each script opens a
session (`SessionAccess`, `Klem`, `DrivingMode`), streams `SpeedMotor` and
`SteerMotor` commands, and closes with `SessionEnd`.

See [`Car/README.md`](../Car/README.md) for the four demos and how to run them.
