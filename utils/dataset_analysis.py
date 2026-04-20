import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_FILE = "results_dataset_summary.json"
SUBJECT_CACHE_FILE = "subject_result.json"


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


def get_hr_status(bpm):
    if bpm <= 0:
        return "No reading"
    elif bpm < 60:
        return "Low heart rate"
    elif bpm <= 100:
        return "Normal heart rate"
    elif bpm <= 120:
        return "Elevated heart rate"
    else:
        return "High alert"


def load_subject_cache(subject_path):
    cache_path = os.path.join(subject_path, SUBJECT_CACHE_FILE)
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None


def save_subject_cache(subject_path, data):
    cache_path = os.path.join(subject_path, SUBJECT_CACHE_FILE)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except:
        pass


def find_video_file(subject_path):
    for f in os.listdir(subject_path):
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return os.path.join(subject_path, f)
    return None


def make_plot(subject_name, plot_dir, times, signal, bpm_over_time, gt_hr, est_hr):
    plot_file = f"{subject_name}_analysis.png"
    plot_path = os.path.join(plot_dir, plot_file)

    # if plot already exists, do not recreate it again
    if os.path.exists(plot_path):
        return plot_file

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

    axs[0, 0].plot(times, signal_arr)
    axs[0, 0].set_title(f"PPG Signal - {subject_name}")
    axs[0, 0].set_xlabel("Time (seconds)")
    axs[0, 0].set_ylabel("Normalized Amplitude")

    if bpm_over_time:
        axs[0, 1].plot(
            [x[0] for x in bpm_over_time],
            [x[1] for x in bpm_over_time]
        )
    axs[0, 1].set_title(f"Heart Rate Over Time - {subject_name}")
    axs[0, 1].set_xlabel("Time (seconds)")
    axs[0, 1].set_ylabel("Heart Rate (BPM)")

    axs[1, 0].bar(["Ground Truth", "Estimated"], [gt_hr, est_hr])
    axs[1, 0].set_title("Heart Rate Comparison")
    axs[1, 0].set_ylabel("Heart Rate (BPM)")

    axs[1, 1].plot(freqs, fft_vals)
    axs[1, 1].set_title("Frequency Spectrum")
    axs[1, 1].set_xlabel("Frequency (Hz)")
    axs[1, 1].set_ylabel("Magnitude")
    axs[1, 1].set_xlim(0, 4)

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)

    return plot_file


def analyze_subject(subject_path, plot_dir, force_refresh=False):
    subject_name = os.path.basename(subject_path)

    # use subject-level cache if available
    if not force_refresh:
        cached = load_subject_cache(subject_path)
        if cached:
            return cached

    video_path = find_video_file(subject_path)

    gt_path1 = os.path.join(subject_path, "ground_truth.txt")
    gt_path2 = os.path.join(subject_path, "groundtruth.txt")

    gt_path = None
    if os.path.exists(gt_path1):
        gt_path = gt_path1
    elif os.path.exists(gt_path2):
        gt_path = gt_path2

    if not video_path or not gt_path:
        return None

    gt_hr = parse_ground_truth(gt_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

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
    processed_frames = 0

    # speed settings
    frame_skip = 2              # process every 2nd frame
    bpm_check_interval = 15     # compute bpm every 15 processed frames
    resize_width = 640          # smaller frame for faster face detection

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_index += 1

        if frame_index % frame_skip != 0:
            continue

        # resize for faster processing
        h, w = frame.shape[:2]
        if w > resize_width:
            scale = resize_width / float(w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)

        for (x, y, fw, fh) in faces:
            fx1 = x + int(fw * 0.30)
            fy1 = y + int(fh * 0.12)
            fx2 = x + int(fw * 0.70)
            fy2 = y + int(fh * 0.28)

            roi = frame[fy1:fy2, fx1:fx2]

            if roi.size != 0:
                green = float(np.mean(roi[:, :, 1]))
                t = frame_index / fps

                signal.append(green)
                times.append(t)
                processed_frames += 1

                if len(signal) >= 150 and processed_frames % bpm_check_interval == 0:
                    est = estimate_bpm_fft(signal[-300:], times[-300:])
                    if est > 0:
                        bpm_over_time.append((t, est))
            break

    cap.release()

    if len(signal) < 150:
        return None

    est_hr = estimate_bpm_fft(signal, times)
    error = round(abs(gt_hr - est_hr), 1) if gt_hr > 0 else 0

    if gt_hr > 0:
        accuracy = round(max(0, 100 - (error / gt_hr * 100)), 1)
    else:
        accuracy = 0

    status = get_hr_status(est_hr)

    plot_file = make_plot(
        subject_name=subject_name,
        plot_dir=plot_dir,
        times=times,
        signal=signal,
        bpm_over_time=bpm_over_time,
        gt_hr=gt_hr,
        est_hr=est_hr
    )

    result = {
        "subject": subject_name,
        "gt_hr": gt_hr,
        "est_hr": est_hr,
        "error": error,
        "accuracy": accuracy,
        "duration": round(times[-1], 1) if times else 0,
        "signal_length": len(signal),
        "plot_file": plot_file,
        "status": status
    }

    save_subject_cache(subject_path, result)
    return result


def run_dataset_analysis(dataset_root, plot_dir, force_refresh=False):
    os.makedirs(plot_dir, exist_ok=True)

    subjects = []
    if os.path.exists(dataset_root):
        for name in sorted(os.listdir(dataset_root)):
            p = os.path.join(dataset_root, name)
            if os.path.isdir(p):
                subjects.append(p)

    results = []
    for subject_path in subjects:
        row = analyze_subject(subject_path, plot_dir, force_refresh=force_refresh)
        if row:
            results.append(row)

    if not results:
        return {
            "success": False,
            "message": "No valid subject folders found"
        }

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

    try:
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None
