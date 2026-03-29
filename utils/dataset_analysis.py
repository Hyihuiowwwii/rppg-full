import os
import cv2
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_video_file(subject_path):
    for file in os.listdir(subject_path):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return os.path.join(subject_path, file)
    return None


def read_ground_truth(gt_path):
    if not os.path.exists(gt_path):
        return 0.0

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


def extract_signal_and_estimate_hr(video_path):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    signal = []
    times = []
    bpm_over_time = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    frame_idx = 0

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
                green_mean = float(np.mean(roi[:, :, 1]))
                t = frame_idx / fps
                signal.append(green_mean)
                times.append(t)

                if len(signal) >= 150:
                    bpm = estimate_bpm(signal[-300:], times[-300:])
                    if bpm > 0:
                        bpm_over_time.append((t, bpm))
            break

        frame_idx += 1

    cap.release()

    if len(signal) < 150:
        return None

    estimated_hr = estimate_bpm(signal, times)
    if estimated_hr <= 0 and bpm_over_time:
        estimated_hr = float(np.mean([x[1] for x in bpm_over_time]))

    return {
        "signal": signal,
        "times": times,
        "estimated_hr": round(float(estimated_hr), 1),
        "bpm_over_time": bpm_over_time,
        "duration": round(times[-1] if times else 0, 1),
        "signal_length": len(signal)
    }


def estimate_bpm(signal, times):
    signal = np.array(signal, dtype=np.float32)
    times = np.array(times, dtype=np.float32)

    if len(signal) < 150:
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
    bpm = peak_freq * 60

    if 40 <= bpm <= 180:
        return bpm
    return 0


def save_subject_plot(subject, result, gt_hr, output_dir):
    signal = np.array(result["signal"], dtype=np.float32)
    signal = signal - np.mean(signal)
    times = np.array(result["times"], dtype=np.float32)

    freqs = []
    fft_vals = []
    if len(signal) > 0 and len(times) > 1:
        duration = times[-1] - times[0]
        fs = len(signal) / duration if duration > 0 else 30
        freqs = np.fft.rfftfreq(len(signal), d=1 / fs)
        fft_vals = np.abs(np.fft.rfft(signal))

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Analysis Plot: {subject}", fontsize=14)

    axs[0, 0].plot(times, signal, color="slateblue")
    axs[0, 0].set_title(f"PPG Signal - {subject}")
    axs[0, 0].set_xlabel("Time (seconds)")
    axs[0, 0].set_ylabel("Normalized Amplitude")

    if result["bpm_over_time"]:
        t_vals = [x[0] for x in result["bpm_over_time"]]
        bpm_vals = [x[1] for x in result["bpm_over_time"]]
        axs[0, 1].plot(t_vals, bpm_vals, color="forestgreen")
    axs[0, 1].set_title(f"Heart Rate Over Time - {subject}")
    axs[0, 1].set_xlabel("Time (seconds)")
    axs[0, 1].set_ylabel("Heart Rate (BPM)")

    axs[1, 0].bar(["Ground Truth", "Estimated"], [gt_hr, result["estimated_hr"]], color=["green", "blue"])
    axs[1, 0].set_title("Heart Rate Comparison")
    axs[1, 0].set_ylabel("Heart Rate (BPM)")

    if len(freqs) > 0:
        axs[1, 1].plot(freqs, fft_vals, color="red")
    axs[1, 1].set_title("Frequency Spectrum")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].set_xlim(0, 4)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{subject}_analysis.png")
    plt.savefig(plot_path)
    plt.close(fig)
    return plot_path


def analyze_dataset(dataset_dir, plot_dir, csv_out):
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results = []
    subject_names = []

    if not os.path.exists(dataset_dir):
        return {"success": False, "message": "Dataset folder not found", "subjects": []}

    for subject in sorted(os.listdir(dataset_dir)):
        subject_path = os.path.join(dataset_dir, subject)
        if not os.path.isdir(subject_path):
            continue

        video_path = find_video_file(subject_path)
        gt_path = os.path.join(subject_path, "ground_truth.txt")

        if not video_path or not os.path.exists(gt_path):
            continue

        gt_hr = read_ground_truth(gt_path)
        analysis = extract_signal_and_estimate_hr(video_path)
        if analysis is None:
            continue

        est_hr = analysis["estimated_hr"]
        error = round(abs(gt_hr - est_hr), 1) if gt_hr > 0 else 0
        accuracy = round(max(0, 100 - (error / gt_hr * 100)), 1) if gt_hr > 0 else 0
        plot_path = save_subject_plot(subject, analysis, gt_hr, plot_dir)

        results.append({
            "subject": subject,
            "gt_hr": gt_hr,
            "est_hr": est_hr,
            "error": error,
            "accuracy": accuracy,
            "duration": analysis["duration"],
            "signal_length": analysis["signal_length"],
            "plot_file": os.path.basename(plot_path)
        })
        subject_names.append(subject)

    if not results:
        return {"success": False, "message": "No valid dataset subjects found", "subjects": []}

    df = pd.DataFrame(results)
    df.to_csv(csv_out, index=False)

    summary = {
        "success": True,
        "total_subjects": len(results),
        "successful_analyses": len(results),
        "average_accuracy": round(float(df["accuracy"].mean()), 1),
        "average_error": round(float(df["error"].mean()), 1),
        "best_accuracy": round(float(df["accuracy"].max()), 1),
        "worst_accuracy": round(float(df["accuracy"].min()), 1),
        "subjects": results
    }
    return summary


def load_dataset_summary():
    csv_out = "results/dataset_summary.csv"
    if not os.path.exists(csv_out):
        return None

    df = pd.read_csv(csv_out)
    if df.empty:
        return None

    return {
        "total_subjects": int(len(df)),
        "successful_analyses": int(len(df)),
        "average_accuracy": round(float(df["accuracy"].mean()), 1),
        "average_error": round(float(df["error"].mean()), 1),
        "best_accuracy": round(float(df["accuracy"].max()), 1),
        "worst_accuracy": round(float(df["accuracy"].min()), 1),
        "subjects": df.to_dict(orient="records")
    }
