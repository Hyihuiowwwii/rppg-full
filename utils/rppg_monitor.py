import cv2
import os
import csv
import time
import numpy as np
from collections import deque
from datetime import datetime

RESULTS_FILE = "results/session_logs.csv"


class RPPGMonitor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.cap = None
        self.mode = "idle"     # idle / live / demo
        self.subject_name = ""
        self.start_time = None
        self.running = False

        self.signal_values = deque(maxlen=300)
        self.signal_times = deque(maxlen=300)
        self.bpm_values = deque(maxlen=200)
        self.bpm_times = deque(maxlen=200)

        self.current_bpm = 0
        self.signal_quality = 0
        self.face_found = False
        self.buffer_target = 150

        os.makedirs("results", exist_ok=True)
        self._ensure_history_file()

    def _ensure_history_file(self):
        if not os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "datetime", "mode", "subject", "avg_bpm",
                    "min_bpm", "max_bpm", "samples", "duration_sec"
                ])

    def reset_signal(self):
        self.signal_values.clear()
        self.signal_times.clear()
        self.bpm_values.clear()
        self.bpm_times.clear()
        self.current_bpm = 0
        self.signal_quality = 0
        self.face_found = False

    def get_demo_subjects(self):
        if not os.path.exists("dataset"):
            return []
        items = []
        for name in os.listdir("dataset"):
            path = os.path.join("dataset", name)
            if os.path.isdir(path):
                items.append(name)
        return sorted(items)

    def _find_video_in_subject(self, subject_path):
        for file in os.listdir(subject_path):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                return os.path.join(subject_path, file)
        return None

    def start_live(self):
        self.stop(save_session=False)
        self.reset_signal()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False, "Webcam not available"
        self.mode = "live"
        self.subject_name = "webcam"
        self.start_time = time.time()
        self.running = True
        return True, "Live monitoring started"

    def start_demo(self, subject):
        self.stop(save_session=False)
        self.reset_signal()

        subject_path = os.path.join("dataset", subject)
        if not os.path.isdir(subject_path):
            return False, "Subject folder not found"

        video_path = self._find_video_in_subject(subject_path)
        if not video_path:
            return False, "No video file found in subject folder"

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            return False, "Demo video failed to open"

        self.mode = "demo"
        self.subject_name = subject
        self.start_time = time.time()
        self.running = True
        return True, f"Demo started: {subject}"

    def stop(self, save_session=True):
        if save_session and self.running:
            self.save_session()

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.running = False
        self.mode = "idle"
        self.subject_name = ""

    def save_session(self):
        bpm_list = list(self.bpm_values)
        duration = round(time.time() - self.start_time, 2) if self.start_time else 0

        avg_bpm = round(sum(bpm_list) / len(bpm_list), 2) if bpm_list else 0
        min_bpm = min(bpm_list) if bpm_list else 0
        max_bpm = max(bpm_list) if bpm_list else 0

        with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.mode,
                self.subject_name,
                avg_bpm,
                min_bpm,
                max_bpm,
                len(self.signal_values),
                duration
            ])

    def process_frame(self, frame, source="live"):
        if source == "live":
            frame = cv2.flip(frame, 1)

        display = frame.copy()
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        self.face_found = False
        status_text = "Face not found"

        for (x, y, w, h) in faces:
            self.face_found = True
            cv2.rectangle(display, (x, y), (x + w, y + h), (50, 220, 80), 2)

            fx1 = x + int(w * 0.30)
            fy1 = y + int(h * 0.12)
            fx2 = x + int(w * 0.70)
            fy2 = y + int(h * 0.28)

            cv2.rectangle(display, (fx1, fy1), (fx2, fy2), (255, 180, 0), 2)

            roi = display[fy1:fy2, fx1:fx2]
            if roi.size != 0:
                green_mean = float(np.mean(roi[:, :, 1]))
                current_time = time.time() - self.start_time

                self.signal_values.append(green_mean)
                self.signal_times.append(current_time)

                if len(self.signal_values) >= self.buffer_target:
                    bpm = self.estimate_bpm_fft()
                    if bpm > 0:
                        self.current_bpm = bpm
                        self.bpm_values.append(bpm)
                        self.bpm_times.append(current_time)

                    self.signal_quality = self.estimate_signal_quality()

            status_text = "Face detected"
            break

        color = (0, 255, 0) if self.face_found else (0, 0, 255)
        cv2.putText(display, f"BPM: {self.current_bpm}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display, status_text, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return display

    def estimate_bpm_fft(self):
        signal = np.array(self.signal_values, dtype=np.float32)
        times = np.array(self.signal_times, dtype=np.float32)

        if len(signal) < self.buffer_target:
            return 0

        signal = signal - np.mean(signal)
        duration = times[-1] - times[0]
        if duration <= 0:
            return 0

        fs = len(signal) / duration
        if fs < 5:
            return 0

        freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
        fft_signal = np.abs(np.fft.rfft(signal))

        valid = (freqs >= 0.8) & (freqs <= 3.0)  # 48 to 180 BPM
        if np.sum(valid) == 0:
            return 0

        peak_freq = freqs[valid][np.argmax(fft_signal[valid])]
        bpm = int(peak_freq * 60)

        if 40 <= bpm <= 180:
            return bpm
        return 0

    def estimate_signal_quality(self):
        if len(self.signal_values) < 40:
            return 0
        signal = np.array(self.signal_values, dtype=np.float32)
        std = np.std(signal)
        quality = min(100, max(0, int(std * 8)))
        return quality

    def generate_frames(self):
        while True:
            if not self.running or self.cap is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "System Idle", (220, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode(".jpg", blank)
                frame_bytes = buffer.tobytes()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
                time.sleep(0.2)
                continue

            success, frame = self.cap.read()

            if not success:
                if self.mode == "demo":
                    # loop demo video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = self.cap.read()
                    if not success:
                        self.stop()
                        continue
                else:
                    self.stop()
                    continue

            source = "live" if self.mode == "live" else "video"
            processed_frame = self.process_frame(frame, source=source)

            ret, buffer = cv2.imencode(".jpg", processed_frame)
            frame_bytes = buffer.tobytes()

            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

    def get_stats(self):
        bpm_list = list(self.bpm_values)

        avg_bpm = round(sum(bpm_list) / len(bpm_list), 2) if bpm_list else 0
        min_bpm = min(bpm_list) if bpm_list else 0
        max_bpm = max(bpm_list) if bpm_list else 0

        duration = round(time.time() - self.start_time, 1) if self.start_time and self.running else 0

        return {
            "mode": self.mode,
            "subject": self.subject_name,
            "current_bpm": self.current_bpm,
            "avg_bpm": avg_bpm,
            "min_bpm": min_bpm,
            "max_bpm": max_bpm,
            "signal_quality": self.signal_quality,
            "buffer_count": len(self.signal_values),
            "buffer_target": self.buffer_target,
            "face_found": self.face_found,
            "status": "Running" if self.running else "Idle",
            "duration": duration,
            "signal_times": list(self.signal_times),
            "signal_values": list(self.signal_values),
            "bpm_times": list(self.bpm_times),
            "bpm_values": list(self.bpm_values)
        }

    def get_history(self, limit=50):
        rows = []
        if not os.path.exists(RESULTS_FILE):
            return rows

        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        rows.reverse()
        return rows[:limit]


monitor = RPPGMonitor()
