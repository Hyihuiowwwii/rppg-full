import cv2
import os
import time
import numpy as np
from collections import deque

try:
    from model.predict import predict_bpm_from_signal
except Exception:
    def predict_bpm_from_signal(signal_window):
        return 0.0


class HeartRateMonitor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.cap = None
        self.mode = "idle"
        self.subject = "-"
        self.use_dl = False
        self.running = False
        self.start_time = None

        self.signal_values = deque(maxlen=300)
        self.signal_times = deque(maxlen=300)
        self.bpm_values = deque(maxlen=120)
        self.bpm_times = deque(maxlen=120)

        self.current_bpm = 0
        self.min_bpm = 0
        self.max_bpm = 0
        self.avg_bpm = 0
        self.buffer_target = 150
        self.signal_quality = 0
        self.status = "Ready to start monitoring"

    def reset(self):
        self.signal_values.clear()
        self.signal_times.clear()
        self.bpm_values.clear()
        self.bpm_times.clear()
        self.current_bpm = 0
        self.min_bpm = 0
        self.max_bpm = 0
        self.avg_bpm = 0
        self.signal_quality = 0
        self.status = "Ready to start monitoring"

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.running = False
        self.mode = "idle"
        self.subject = "-"
        self.use_dl = False
        self.status = "Stopped"

    def get_demo_subjects(self, dataset_root):
        if not os.path.exists(dataset_root):
            return []
        return sorted([x for x in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, x))])

    def start_live(self):
        self.stop()
        self.reset()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            return False, "Webcam not available"

        self.mode = "live"
        self.subject = "webcam"
        self.running = True
        self.start_time = time.time()
        self.status = "Waiting for signal..."
        return True, "Live monitoring started"

    def start_demo(self, dataset_root, subject, use_dl=False):
        self.stop()
        self.reset()

        subject_path = os.path.join(dataset_root, subject)
        if not os.path.isdir(subject_path):
            return False, "Subject not found"

        video_path = None
        for f in os.listdir(subject_path):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(subject_path, f)
                break

        if video_path is None:
            return False, "Demo video not found"

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            return False, "Could not open demo video"

        self.mode = "demo"
        self.subject = subject
        self.use_dl = use_dl
        self.running = True
        self.start_time = time.time()
        self.status = "Running demo mode"
        return True, "Demo mode started"

    def estimate_fft_bpm(self, signal, times):
        signal = np.array(signal, dtype=np.float32)
        times = np.array(times, dtype=np.float32)

        if len(signal) < 120:
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

        valid = (freqs >= 0.8) & (freqs <= 3.0)
        if np.sum(valid) == 0:
            return 0

        peak_freq = freqs[valid][np.argmax(fft_signal[valid])]
        bpm = int(peak_freq * 60)

        if 40 <= bpm <= 180:
            return bpm
        return 0

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        face_found = False

        for (x, y, w, h) in faces:
            face_found = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            fx1 = x + int(w * 0.30)
            fy1 = y + int(h * 0.12)
            fx2 = x + int(w * 0.70)
            fy2 = y + int(h * 0.28)

            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)

            roi = frame[fy1:fy2, fx1:fx2]
            if roi.size != 0:
                green_mean = float(np.mean(roi[:, :, 1]))
                current_time = time.time() - self.start_time

                self.signal_values.append(green_mean)
                self.signal_times.append(current_time)

                if len(self.signal_values) >= self.buffer_target:
                    bpm = 0

                    if self.use_dl:
                        bpm = int(predict_bpm_from_signal(list(self.signal_values)))
                        if bpm <= 0:
                            bpm = self.estimate_fft_bpm(self.signal_values, self.signal_times)
                    else:
                        bpm = self.estimate_fft_bpm(self.signal_values, self.signal_times)

                    if bpm > 0:
                        self.current_bpm = bpm
                        self.bpm_values.append(bpm)
                        self.bpm_times.append(current_time)

                        self.min_bpm = min(self.bpm_values)
                        self.max_bpm = max(self.bpm_values)
                        self.avg_bpm = round(sum(self.bpm_values) / len(self.bpm_values), 1)

                    self.signal_quality = min(100, int((len(self.signal_values) / self.buffer_target) * 100))
                    self.status = "Calculating heart rate..." if self.current_bpm == 0 else "Heart rate detected"
                else:
                    self.signal_quality = int((len(self.signal_values) / self.buffer_target) * 100)
                    self.status = "Collecting data..."

            break

        if not face_found:
            cv2.putText(frame, "Face not detected - Please position face in frame", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, f"Buffer: {len(self.signal_values)}/{self.buffer_target}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame

    def generate_frames(self):
        while True:
            if not self.running or self.cap is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Camera Feed Will Appear Here", (130, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                ret, buffer = cv2.imencode(".jpg", blank)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
                time.sleep(0.2)
                continue

            success, frame = self.cap.read()

            if not success:
                if self.mode == "demo":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    self.stop()
                    continue

            if self.mode == "live":
                frame = cv2.flip(frame, 1)

            processed = self.process_frame(frame)

            ret, buffer = cv2.imencode(".jpg", processed)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    def get_stats(self):
        return {
            "mode": self.mode,
            "subject": self.subject,
            "current_bpm": self.current_bpm,
            "min_bpm": self.min_bpm,
            "max_bpm": self.max_bpm,
            "avg_bpm": self.avg_bpm,
            "signal_quality": self.signal_quality,
            "buffer_count": len(self.signal_values),
            "buffer_target": self.buffer_target,
            "status": self.status,
            "signal_values": list(self.signal_values),
            "signal_times": list(self.signal_times),
            "bpm_values": list(self.bpm_values),
            "bpm_times": list(self.bpm_times)
        }


monitor = HeartRateMonitor()
