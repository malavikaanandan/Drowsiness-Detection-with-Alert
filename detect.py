"""Real-time drowsiness and yawn detection with an audible alarm.

Watches the eyes and mouth through a webcam using dlib's 68-point facial
landmarks:
  * Eyes closed (low Eye Aspect Ratio) for too many frames -> DROWSINESS alert
  * Mouth wide open (high Mouth Aspect Ratio) for too many frames -> YAWN warning
Blinks and yawns are counted on screen, and every alert can be saved to a CSV log.

Run `python detect.py --help` to see all options.
"""
import argparse
import csv
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from scipy.spatial import distance

HERE = os.path.dirname(os.path.abspath(__file__))

# Landmark index ranges in dlib's 68-point model (end index is exclusive)
LEFT_EYE = (42, 48)
RIGHT_EYE = (36, 42)
INNER_MOUTH = (60, 68)


# ---------------------------------------------------------------- geometry
def eye_aspect_ratio(eye):
    """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) for the 6 eye points."""
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c) if c else 0.0


def mouth_aspect_ratio(mouth):
    """MAR for the 8 inner-mouth points (60-67): mouth height over width."""
    a = distance.euclidean(mouth[1], mouth[7])  # 61-67
    b = distance.euclidean(mouth[2], mouth[6])  # 62-66
    c = distance.euclidean(mouth[3], mouth[5])  # 63-65
    d = distance.euclidean(mouth[0], mouth[4])  # 60-64
    return (a + b + c) / (3.0 * d) if d else 0.0


def shape_to_np(shape):
    """Convert a dlib full_object_detection to a (68, 2) NumPy array."""
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(shape.num_parts)])


# ------------------------------------------------------------------ state
class FatigueMonitor:
    """Turns per-frame EAR/MAR values into blink, drowsiness and yawn events.

    Kept free of camera/dlib code so it can be tested on its own.
    """

    def __init__(self, ear_thresh=0.25, ear_frames=20, mar_thresh=0.6, yawn_frames=15):
        self.ear_thresh = ear_thresh
        self.ear_frames = ear_frames
        self.mar_thresh = mar_thresh
        self.yawn_frames = yawn_frames
        self.closed_frames = 0
        self.open_mouth_frames = 0
        self.blinks = 0
        self.yawns = 0
        self.drowsy_events = 0
        self.drowsy = False
        self.yawning = False

    def update(self, ear, mar):
        """Feed one frame. Returns a list of new events ('drowsy', 'yawn')."""
        events = []

        if ear < self.ear_thresh:
            self.closed_frames += 1
            if self.closed_frames >= self.ear_frames and not self.drowsy:
                self.drowsy = True
                self.drowsy_events += 1
                events.append("drowsy")
        else:
            # a short closure that never became drowsiness is a blink
            if 2 <= self.closed_frames < self.ear_frames:
                self.blinks += 1
            self.closed_frames = 0
            self.drowsy = False

        if mar > self.mar_thresh:
            self.open_mouth_frames += 1
            if self.open_mouth_frames >= self.yawn_frames and not self.yawning:
                self.yawning = True
                self.yawns += 1
                events.append("yawn")
        else:
            self.open_mouth_frames = 0
            self.yawning = False

        return events


# ------------------------------------------------------------------ alarm
class Alarm:
    """Plays the alarm sound in the background, never more than one at a time."""

    def __init__(self, path):
        self.path = path
        self._thread = None

    def _play(self):
        try:
            import playsound
            playsound.playsound(self.path)
        except Exception:
            if sys.platform == "win32":
                import winsound
                winsound.PlaySound(self.path, winsound.SND_FILENAME)
            else:
                print("\a", end="", flush=True)

    def trigger(self):
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._play, daemon=True)
            self._thread.start()


# -------------------------------------------------------------------- log
class EventLog:
    """Appends alert events to a CSV file (timestamp, event, ear, mar)."""

    def __init__(self, path):
        self.path = path
        if path:
            new = not os.path.exists(path)
            self._file = open(path, "a", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            if new:
                self._writer.writerow(["timestamp", "event", "ear", "mar"])

    def write(self, event, ear, mar):
        if self.path:
            self._writer.writerow([datetime.now().isoformat(timespec="seconds"),
                                   event, f"{ear:.3f}", f"{mar:.3f}"])
            self._file.flush()

    def close(self):
        if self.path:
            self._file.close()


# ------------------------------------------------------------------- draw
def draw_hud(frame, ear, mar, monitor, fps):
    h, w = frame.shape[:2]
    lines = [
        f"EAR: {ear:.2f}  (closed < {monitor.ear_thresh})",
        f"MAR: {mar:.2f}  (yawn > {monitor.mar_thresh})",
        f"Blinks: {monitor.blinks}   Yawns: {monitor.yawns}",
        f"Drowsy alerts: {monitor.drowsy_events}   FPS: {fps:.0f}",
    ]
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (10, h - 75 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if monitor.drowsy:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
        cv2.putText(frame, "DROWSINESS ALERT!", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    elif monitor.yawning:
        cv2.putText(frame, "Yawning - take a break?", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


def resize(frame, width):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (width, int(h * width / w)))


# ------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Real-time drowsiness and yawn detection.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--model", default=os.path.join(HERE, "shape_predictor_68_face_landmarks.dat"),
                   help="path to dlib's 68-point landmark model")
    p.add_argument("--alarm", default=os.path.join(HERE, "alarm.wav"), help="alarm sound file")
    p.add_argument("--ear-thresh", type=float, default=0.25, help="EAR below this = eyes closed")
    p.add_argument("--ear-frames", type=int, default=20, help="closed frames before drowsiness alert")
    p.add_argument("--mar-thresh", type=float, default=0.6, help="MAR above this = mouth wide open")
    p.add_argument("--yawn-frames", type=int, default=15, help="open-mouth frames to count a yawn")
    p.add_argument("--log", default=None, help="save alert events to this CSV file")
    p.add_argument("--width", type=int, default=640, help="processing width in pixels")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(args.model):
        sys.exit(f"Landmark model not found: {args.model}\n"
                 "Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 "
                 "and extract it next to detect.py (see README).")

    import dlib  # imported here so the helpers above can be used without dlib

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.model)
    monitor = FatigueMonitor(args.ear_thresh, args.ear_frames, args.mar_thresh, args.yawn_frames)
    alarm = Alarm(args.alarm)
    log = EventLog(args.log)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}.")

    print("Monitoring started. Press 'q' to quit.")
    start = time.time()
    prev = start
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Camera stopped sending frames.")
                break
            frame = resize(frame, args.width)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            faces = detector(gray, 0)
            ear = mar = 0.0
            if faces:
                # track only the largest face, i.e. the driver
                face = max(faces, key=lambda r: r.width() * r.height())
                pts = shape_to_np(predictor(gray, face))
                left, right = pts[slice(*LEFT_EYE)], pts[slice(*RIGHT_EYE)]
                mouth = pts[slice(*INNER_MOUTH)]
                ear = (eye_aspect_ratio(left) + eye_aspect_ratio(right)) / 2.0
                mar = mouth_aspect_ratio(mouth)

                for part in (left, right, mouth):
                    cv2.drawContours(frame, [cv2.convexHull(part)], -1, (0, 255, 0), 1)

                for event in monitor.update(ear, mar):
                    log.write(event, ear, mar)
                    if event == "drowsy":
                        print(f"[{datetime.now():%H:%M:%S}] Drowsiness detected!")
                    else:
                        print(f"[{datetime.now():%H:%M:%S}] Yawn detected.")
                if monitor.drowsy:
                    alarm.trigger()
            else:
                cv2.putText(frame, "No face detected", (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            draw_hud(frame, ear, mar, monitor, fps)
            cv2.imshow("Drowsiness Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        log.close()

    minutes = (time.time() - start) / 60
    print(f"\nSession summary ({minutes:.1f} min): {monitor.blinks} blinks, "
          f"{monitor.yawns} yawns, {monitor.drowsy_events} drowsiness alerts.")


if __name__ == "__main__":
    main()
