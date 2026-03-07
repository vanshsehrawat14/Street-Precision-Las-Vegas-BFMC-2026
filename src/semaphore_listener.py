#!/usr/bin/env python3
"""
semaphore_listener.py - BFMC V2X Traffic Light Receiver
Listens on UDP port 5007 for semaphore state broadcasts.
State: 0=RED, 1=YELLOW, 2=GREEN
Broadcasts at 5Hz from the BFMC track infrastructure.
"""
import socket
import json
import threading
import rospy
from std_msgs.msg import String


class SemaphoreListener:
    def __init__(self):
        self.semaphores = {}   # {semaphore_id: {"state": 0/1/2, "x": float, "y": float}}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.pub = rospy.Publisher("/v2x/semaphore", String, queue_size=10)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()
        rospy.loginfo("[V2X] Semaphore listener started on UDP port 5007")

    def stop(self):
        self._running = False

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 5007))
        sock.settimeout(1.0)
        while self._running:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                sem_id = msg.get("id", "unknown")
                state  = int(msg.get("state", 0))
                x      = float(msg.get("x", 0.0))
                y      = float(msg.get("y", 0.0))
                with self._lock:
                    self.semaphores[sem_id] = {"state": state, "x": x, "y": y}
                self.pub.publish(String(data=json.dumps({"id": sem_id, "state": state})))
                rospy.logdebug_throttle(2.0, f"[V2X] Semaphore {sem_id}: state={state} ({'RED' if state==0 else 'YELLOW' if state==1 else 'GREEN'})")
            except socket.timeout:
                continue
            except Exception as e:
                rospy.logwarn_throttle(5.0, f"[V2X] Parse error: {e}")
        sock.close()

    def get_nearest_state(self, car_x=0.0, car_y=0.0):
        """
        Returns state of the nearest semaphore to the car's position.
        Returns None if no semaphores received yet.
        State: 0=RED, 1=YELLOW, 2=GREEN
        """
        with self._lock:
            if not self.semaphores:
                return None
            nearest = min(
                self.semaphores.values(),
                key=lambda s: (s["x"] - car_x)**2 + (s["y"] - car_y)**2
            )
            return nearest["state"]

    def get_all(self):
        with self._lock:
            return dict(self.semaphores)

    @staticmethod
    def state_to_color(state):
        return {0: "red", 1: "yellow", 2: "green"}.get(state, None)
