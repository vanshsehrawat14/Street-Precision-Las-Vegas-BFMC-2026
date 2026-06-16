#!/usr/bin/env python3
"""
Finite-state machine and detection parsing for the BFMC pipeline.

Consumed by lane_follow_bfmc.py. Raw Roboflow predictions are turned into a
structured Detection (parse_predictions); the FSM advances one step per frame
(update_fsm) and returns a driving action together with a speed scale:

    action       one of "NORMAL", "SLOW_DOWN", "STOP"
    speed_scale  1.0 = full speed, 0.0 = stopped

States cover stop signs, traffic lights, and pedestrians/crosswalks.
draw_fsm_overlay renders the current state and detections onto a frame.
"""

import cv2
import numpy as np
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional


# FSM states

class State(Enum):
    DRIVE           = auto()   # Normal lane following
    APPROACHING_STOP= auto()   # Stop sign visible, slowing down
    STOPPED_AT_SIGN = auto()   # Fully stopped, waiting 2s
    RESUMING        = auto()   # Just left stop, ignoring sign for 5s
    PEDESTRIAN_WAIT = auto()   # Pedestrian/crosswalk detected, waiting
    TRAFFIC_RED     = auto()   # Red light, stopped
    TRAFFIC_GREEN   = auto()   # Green light seen, proceeding
    PARKING         = auto()   # Parking maneuver (future)
    EMERGENCY_STOP  = auto()   # Obstacle too close (existing logic)


# Detection result

@dataclass
class Detection:
    classes: set = field(default_factory=set)
    traffic_light_color: Optional[str] = None   # "red", "green", "yellow", None
    stop_sign_confidence: float = 0.0
    pedestrian_confidence: float = 0.0
    raw_predictions: list = field(default_factory=list)


# FSM context

@dataclass
class BFMCStateMachine:
    state: State = State.DRIVE
    state_entered_at: float = 0.0        # rospy.Time.now().to_sec()
    ignore_stop_until: float = 0.0       # ignore stop sign after resuming
    ped_clear_count: int = 0             # consecutive frames with no pedestrian
    ped_clear_needed: int = 8            # frames needed before resuming
    stop_hold_secs: float = 2.0          # how long to hold at stop sign
    approach_speed_scale: float = 0.5   # slow to 50% when approaching


# Traffic light color detector

def detect_traffic_light_color(bgr_crop: np.ndarray) -> Optional[str]:
    """
    Given a cropped BGR image of a traffic light bounding box,
    return 'red', 'green', 'yellow', or None.
    Uses HSV color analysis of the top/middle/bottom thirds.
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    if h < 9:
        return None

    third = h // 3
    regions = {
        "red":    hsv[0:third, :, :],
        "yellow": hsv[third:2*third, :, :],
        "green":  hsv[2*third:, :, :],
    }

    # HSV ranges
    masks = {}
    # Red wraps around hue 0/180
    r1 = cv2.inRange(regions["red"],    np.array([0, 100, 100]),  np.array([10, 255, 255]))
    r2 = cv2.inRange(regions["red"],    np.array([170, 100, 100]),np.array([180, 255, 255]))
    masks["red"]    = cv2.bitwise_or(r1, r2)
    masks["yellow"] = cv2.inRange(regions["yellow"], np.array([15, 100, 100]), np.array([35, 255, 255]))
    masks["green"]  = cv2.inRange(regions["green"],  np.array([40, 50, 50]),  np.array([90, 255, 255]))

    scores = {color: float(np.sum(mask > 0)) for color, mask in masks.items()}
    best = max(scores, key=scores.get)

    # Only return if there's a meaningful signal
    area = bgr_crop.shape[0] * bgr_crop.shape[1]
    if scores[best] < area * 0.05:
        return None

    return best


# Parse raw Roboflow predictions

def parse_predictions(raw_predictions: list, frame: np.ndarray) -> Detection:
    """Convert Roboflow predictions into a structured Detection object."""
    det = Detection(raw_predictions=raw_predictions)

    for pred in raw_predictions:
        cls = pred.get("class", "")
        conf = pred.get("confidence", 0.0)
        det.classes.add(cls)

        if cls == "stop_sign":
            det.stop_sign_confidence = max(det.stop_sign_confidence, conf)

        if cls in ("pedestrian", "person"):
            det.pedestrian_confidence = max(det.pedestrian_confidence, conf)

        # For traffic lights, crop the bounding box and analyze color
        if cls == "traffic_light" and frame is not None:
            h, w = frame.shape[:2]
            cx = int(pred.get("x", 0))
            cy = int(pred.get("y", 0))
            bw = int(pred.get("width", 0))
            bh = int(pred.get("height", 0))
            x1 = max(0, cx - bw // 2)
            y1 = max(0, cy - bh // 2)
            x2 = min(w, cx + bw // 2)
            y2 = min(h, cy + bh // 2)
            crop = frame[y1:y2, x1:x2]
            color = detect_traffic_light_color(crop)
            if color:
                det.traffic_light_color = color
                det.classes.add(f"traffic_light_{color}")

    return det


# FSM transition logic

def update_fsm(fsm: BFMCStateMachine, det: Detection,
               now_sec: float) -> tuple:
    """
    Update FSM given a Detection and current time.
    Returns (action_str, speed_scale, updated_fsm).

    action_str:  "NORMAL" | "SLOW_DOWN" | "STOP"
    speed_scale: 1.0 = full speed, 0.5 = half, 0.0 = stopped
    """
    s = fsm.state

    # DRIVE
    if s == State.DRIVE:
        # Stop sign visible and not in ignore window
        if "stop_sign" in det.classes and now_sec > fsm.ignore_stop_until:
            fsm.state = State.APPROACHING_STOP
            fsm.state_entered_at = now_sec
            return "SLOW_DOWN", fsm.approach_speed_scale, fsm

        # Red traffic light
        if det.traffic_light_color == "red":
            fsm.state = State.TRAFFIC_RED
            fsm.state_entered_at = now_sec
            return "STOP", 0.0, fsm

        # Pedestrian or crosswalk
        if "pedestrian" in det.classes or "person" in det.classes or "crosswalk" in det.classes:
            fsm.state = State.PEDESTRIAN_WAIT
            fsm.state_entered_at = now_sec
            fsm.ped_clear_count = 0
            return "SLOW_DOWN", 0.3, fsm

        return "NORMAL", 1.0, fsm

    # APPROACHING_STOP
    elif s == State.APPROACHING_STOP:
        if "stop_sign" in det.classes and now_sec > fsm.ignore_stop_until:
            # Still seeing sign, keep slowing
            return "SLOW_DOWN", fsm.approach_speed_scale, fsm
        else:
            # Sign disappeared (car is at line) or ignore window: full stop
            fsm.state = State.STOPPED_AT_SIGN
            fsm.state_entered_at = now_sec
            return "STOP", 0.0, fsm

    # STOPPED_AT_SIGN
    elif s == State.STOPPED_AT_SIGN:
        elapsed = now_sec - fsm.state_entered_at
        if elapsed >= fsm.stop_hold_secs:
            # Done waiting; resume and ignore stop sign for 5 seconds
            fsm.state = State.RESUMING
            fsm.state_entered_at = now_sec
            fsm.ignore_stop_until = now_sec + 5.0
            return "NORMAL", 1.0, fsm
        return "STOP", 0.0, fsm

    # RESUMING
    elif s == State.RESUMING:
        # Back to normal driving after leaving stop sign
        fsm.state = State.DRIVE
        return "NORMAL", 1.0, fsm

    # TRAFFIC_RED
    elif s == State.TRAFFIC_RED:
        if det.traffic_light_color == "green":
            fsm.state = State.TRAFFIC_GREEN
            fsm.state_entered_at = now_sec
            return "NORMAL", 1.0, fsm
        if det.traffic_light_color != "red" and det.traffic_light_color is None:
            # Light disappeared, wait a moment then go
            elapsed = now_sec - fsm.state_entered_at
            if elapsed > 3.0:
                fsm.state = State.DRIVE
                return "NORMAL", 1.0, fsm
        return "STOP", 0.0, fsm

    # TRAFFIC_GREEN
    elif s == State.TRAFFIC_GREEN:
        fsm.state = State.DRIVE
        return "NORMAL", 1.0, fsm

    # PEDESTRIAN_WAIT
    elif s == State.PEDESTRIAN_WAIT:
        ped_present = ("pedestrian" in det.classes or
                       "person" in det.classes or
                       "crosswalk" in det.classes)
        if ped_present:
            fsm.ped_clear_count = 0
            return "STOP", 0.0, fsm
        else:
            fsm.ped_clear_count += 1
            if fsm.ped_clear_count >= fsm.ped_clear_needed:
                fsm.state = State.DRIVE
                return "NORMAL", 1.0, fsm
            return "SLOW_DOWN", 0.3, fsm

    # Fallback
    return "NORMAL", 1.0, fsm


# Debug overlay

STATE_COLORS = {
    State.DRIVE:            (0, 255, 0),
    State.APPROACHING_STOP: (0, 165, 255),
    State.STOPPED_AT_SIGN:  (0, 0, 255),
    State.RESUMING:         (255, 255, 0),
    State.PEDESTRIAN_WAIT:  (255, 0, 255),
    State.TRAFFIC_RED:      (0, 0, 255),
    State.TRAFFIC_GREEN:    (0, 255, 0),
    State.PARKING:          (255, 165, 0),
    State.EMERGENCY_STOP:   (0, 0, 255),
}

def draw_fsm_overlay(frame: np.ndarray, fsm: BFMCStateMachine,
                     action: str, speed_scale: float,
                     detections: list) -> np.ndarray:
    """Draw FSM state, action, detections on frame."""
    dbg = frame.copy()
    color = STATE_COLORS.get(fsm.state, (255, 255, 255))

    cv2.putText(dbg, f"FSM: {fsm.state.name}", (10, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(dbg, f"ACT: {action}  spd_scale={speed_scale:.1f}", (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)

    for pred in detections:
        cls  = pred.get("class", "")
        conf = pred.get("confidence", 0.0)
        cx   = int(pred.get("x", 0))
        cy   = int(pred.get("y", 0))
        bw   = int(pred.get("width", 0))
        bh   = int(pred.get("height", 0))
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2
        cv2.rectangle(dbg, (x1, y1), (x2, y2), color, 2)
        cv2.putText(dbg, f"{cls} {conf:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return dbg
