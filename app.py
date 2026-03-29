from flask import Flask, render_template, Response, jsonify, request
import os
from utils.rppg_monitor import monitor
from utils.dataset_analysis import analyze_dataset, load_dataset_summary

app = Flask(__name__)

os.makedirs("results", exist_ok=True)
os.makedirs("static/plots", exist_ok=True)

@app.route("/")
def dashboard():
    history = monitor.get_history(limit=10)
    last = history[0] if history else None

    total_sessions = len(history)
    avg_bpm = round(
        sum(float(x["avg_bpm"]) for x in history if float(x["avg_bpm"]) > 0) /
        max(1, len([x for x in history if float(x["avg_bpm"]) > 0])),
        2
    ) if history else 0

    return render_template(
        "dashboard.html",
        total_sessions=total_sessions,
        avg_bpm=avg_bpm,
        last=last
    )

@app.route("/realtime")
def realtime():
    subjects = monitor.get_demo_subjects()
    return render_template("realtime.html", subjects=subjects)

@app.route("/dataset-analysis")
def dataset_analysis_page():
    summary = load_dataset_summary()
    return render_template("dataset_analysis.html", summary=summary)

@app.route("/model-info")
def model_info():
    return render_template("model_info.html")

@app.route("/history")
def history():
    rows = monitor.get_history(limit=100)
    return render_template("history.html", rows=rows)

@app.route("/api/start_live", methods=["POST"])
def start_live():
    ok, msg = monitor.start_live()
    return jsonify({"success": ok, "message": msg})

@app.route("/api/start_demo", methods=["POST"])
def start_demo():
    data = request.get_json()
    subject = data.get("subject", "")
    ok, msg = monitor.start_demo(subject)
    return jsonify({"success": ok, "message": msg})

@app.route("/api/stop", methods=["POST"])
def stop():
    monitor.stop()
    return jsonify({"success": True})

@app.route("/api/stats")
def stats():
    return jsonify(monitor.get_stats())

@app.route("/api/history")
def api_history():
    return jsonify(monitor.get_history(limit=50))

@app.route("/video_feed")
def video_feed():
    return Response(
        monitor.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/api/run_dataset_analysis", methods=["POST"])
def run_dataset_analysis():
    result = analyze_dataset("dataset", "static/plots", "results/dataset_summary.csv")
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
