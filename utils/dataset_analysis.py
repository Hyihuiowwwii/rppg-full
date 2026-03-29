import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_FILE = "results_dataset_summary.json"


def parse_ground_truth(gt_path):
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().replace(",", " ").replace("\n", " ")

    values = []
    for token in text.split():
        try:
            val = float(token)
            if 40 <= val <= 180:
                values.append(val)
        except:
            pass

    if not values:
        return 0.0

    return round(float(np.mean(values)), 1)


def estimate_bpm_fft(signal, times):
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
    bpm = float(peak_freq * 60)

    if 40 <= bpm <= 180:
        return round(bpm, 1)
    return 0


def analyze_subject(subject_path, plot_dir):
    subject_name = os.path.basename(subject_path)

    video_path = None
    for f in os.listdir(subject_path):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(subject_path, f)
            break

    gt_path = os.path.join(subject_path, "ground_truth.txt")
    if not video_path or not os.path.exists(gt_path):
        return None

    gt_hr = parse_ground_truth(gt_path)

    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    signal = []
    times = []
    bpm_over_time = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            fx1 = x + int(w * 0.30)
            fy1 = y + int(h * 0.12)
            fx2 = x + int(w * 0.70)
            fy2 = y + int(h * 0.28)

            roi = frame[fy1:fy2, fx1:fx2]
            if roi.size != 0:
                green = float(np.mean(roi[:, :, 1]))
                t = frame_index / fps
                signal.append(green)
                times.append(t)

                if len(signal) >= 150:
                    est = estimate_bpm_fft(signal[-300:], times[-300:])
                    if est > 0:
                        bpm_over_time.append((t, est))
            break

        frame_index += 1

    cap.release()

    if len(signal) < 150:
        return None

    est_hr = estimate_bpm_fft(signal, times)
    error = round(abs(gt_hr - est_hr), 1) if gt_hr > 0 else 0
    accuracy = round(max(0, 100 - (error / gt_hr * 100)), 1) if gt_hr > 0 else 0

    # plots
    signal_arr = np.array(signal, dtype=np.float32)
    signal_arr = signal_arr - np.mean(signal_arr)

    freqs = []
    fft_vals = []
    if len(signal_arr) > 0 and len(times) > 1:
        duration = times[-1] - times[0]
        fs = len(signal_arr) / duration if duration > 0 else 30
        freqs = np.fft.rfftfreq(len(signal_arr), d=1 / fs)
        fft_vals = np.abs(np.fft.rfft(signal_arr))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Analysis Plot: {subject_name}", fontsize=14)

    axs[0, 0].plot(times, signal_arr, color="slateblue")
    axs[0, 0].set_title(f"PPG Signal - {subject_name}")
    axs[0, 0].set_xlabel("Time (seconds)")
    axs[0, 0].set_ylabel("Normalized Amplitude")

    if bpm_over_time:
        axs[0, 1].plot([x[0] for x in bpm_over_time], [x[1] for x in bpm_over_time], color="green")
    axs[0, 1].set_title(f"Heart Rate Over Time - {subject_name}")
    axs[0, 1].set_xlabel("Time (seconds)")
    axs[0, 1].set_ylabel("Heart Rate (BPM)")

    axs[1, 0].bar(["Ground Truth", "Estimated"], [gt_hr, est_hr], color=["green", "blue"])
    axs[1, 0].set_title("Heart Rate Comparison")
    axs[1, 0].set_ylabel("Heart Rate (BPM)")

    axs[1, 1].plot(freqs, fft_vals, color="red")
    axs[1, 1].set_title("Frequency Spectrum")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].set_xlim(0, 4)

    plot_file = f"{subject_name}_analysis.png"
    plot_path = os.path.join(plot_dir, plot_file)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

    return {
        "subject": subject_name,
        "gt_hr": gt_hr,
        "est_hr": est_hr,
        "error": error,
        "accuracy": accuracy,
        "duration": round(times[-1], 1) if times else 0,
        "signal_length": len(signal),
        "plot_file": plot_file
    }


def run_dataset_analysis(dataset_root, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    subjects = []
    if os.path.exists(dataset_root):
        for name in sorted(os.listdir(dataset_root)):
            p = os.path.join(dataset_root, name)
            if os.path.isdir(p):
                subjects.append(p)

    results = []
    for subject_path in subjects:
        row = analyze_subject(subject_path, plot_dir)
        if row:
            results.append(row)

    if not results:
        return {"success": False, "message": "No valid subject folders found"}

    avg_accuracy = round(sum(x["accuracy"] for x in results) / len(results), 1)
    avg_error = round(sum(x["error"] for x in results) / len(results), 1)
    best_accuracy = round(max(x["accuracy"] for x in results), 1)
    worst_accuracy = round(min(x["accuracy"] for x in results), 1)

    summary = {
        "success": True,
        "total_subjects": len(results),
        "successful_analyses": len(results),
        "average_accuracy": avg_accuracy,
        "average_error": avg_error,
        "best_accuracy": best_accuracy,
        "worst_accuracy": worst_accuracy,
        "subjects": results
    }

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def load_latest_dataset_summary():
    if not os.path.exists(SUMMARY_FILE):
        return None
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
