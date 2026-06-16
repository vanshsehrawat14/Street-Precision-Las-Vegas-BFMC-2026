# Street Precision Las Vegas, BFMC 2026

Autonomous driving software for our entry in the
[Bosch Future Mobility Challenge](https://boschfuturemobility.com/) 2026, a
1:10-scale self-driving car competition on a model city track.

The repository has two targets that share one control approach: a ROS pipeline
for the BFMC simulator, and Jetson scripts that drive the physical car. Both
turn a single forward camera into a speed and a steering command every frame.

## What it does

- **Lane following.** White-lane detection with a sliding-window polynomial fit
  and Stanley steering from cross-track and heading error.
- **Obstacle handling.** Motion-based detection in a front region of interest, a
  lane-offset avoidance maneuver, and an emergency stop when an obstacle is close
  and centered.
- **Traffic rules.** Roboflow YOLO detection drives a finite-state machine for
  stop signs, traffic lights, and pedestrians.
- **V2X.** A UDP listener for the track's semaphore broadcasts, used as a
  fallback for traffic-light state.

## Repository layout

```
Car/          Jetson scripts that drive the physical car over the Pi Brain's
              Socket.IO link. Start here for hardware demos.
Simulation/   ROS pipeline (lane following, obstacle avoidance, FSM, V2X) for
              the BFMC simulator.
docs/         Architecture notes, project plan, and status reports.
```

- [`Car/README.md`](Car/README.md): the four physical-car demos.
- [`Simulation/README.md`](Simulation/README.md): the ROS pipeline.
- [`docs/architecture.md`](docs/architecture.md): hardware and software overview.
- [`docs/status-reports/`](docs/status-reports/): progress reports and demo videos.

## Getting started

```bash
pip install -r requirements.txt
```

The Simulation pipeline also needs ROS 1 (Noetic), which provides `rospy` and
`cv_bridge`. Detection uses the Roboflow inference API. Copy `.env.example` to
`.env` and set `ROBOFLOW_API_KEY`.

## Team

Street Precision Las Vegas.
