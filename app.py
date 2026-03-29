from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
import os
from utils.db import init_db, register_user, check_login, get_user, save_session_log, get_session_logs
from utils.monitor import monitor
from utils.dataset_analysis import run_dataset_analysis, load_latest_dataset_summary

app = Flask(__name__)
app.secret_key = "rppg-secret-key"

os.makedirs("static/plots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

init_db()


def login_required():
    return "user_id" in session


@app.route("/")
def home():
    if login_required():
        return redirect(url_for("dashboard"))
    return render_template("home.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip()
        password = request.form["password"].strip()

        ok, msg = register_user(username, email, password)
        if ok:
            flash("Registration successful. Please login.")
            return redirect(url_for("login"))
        flash(msg)
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"].strip()

        user = check_login(username, password)
        if user:
            session["user_id"] = user["id"]
            flash("Login successful!")
            return redirect(url_for("dashboard"))
        flash("Invalid username or password")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully")
    return redirect(url_for("home"))


@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    history = get_session_logs(limit=10)
    dataset_summary = load_latest_dataset_summary()

    return render_template(
        "dashboard.html",
        user=user,
        history=history,
        dataset_summary=dataset_summary
    )


@app.route("/real-time-monitoring")
def realtime_monitoring():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    subjects = monitor.get_demo_subjects("dataset")
    return render_template("realtime.html", user=user, subjects=subjects)


@app.route("/dataset-analysis")
def dataset_analysis():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    summary = load_latest_dataset_summary()
    return render_template("dataset_analysis.html", user=user, summary=summary)


@app.route("/model-info")
def model_info():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    summary = load_latest_dataset_summary()
    return render_template("model_info.html", user=user, summary=summary)


@app.route("/history")
def history():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    logs = get_session_logs(limit=100)
    return render_template("history.html", user=user, logs=logs)


@app.route("/video_feed")
def video_feed():
    return Response(
        monitor.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/start_live", methods=["POST"])
def api_start_live():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    ok, msg = monitor.start_live()
    return jsonify({"success": ok, "message": msg})


@app.route("/api/start_demo", methods=["POST"])
def api_start_demo():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    subject = request.json.get("subject", "")
    use_dl = bool(request.json.get("use_dl", False))
    ok, msg = monitor.start_demo("dataset", subject, use_dl=use_dl)
    return jsonify({"success": ok, "message": msg})


@app.route("/api/stop_monitor", methods=["POST"])
def api_stop_monitor():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    stats = monitor.get_stats()
    if stats["avg_bpm"] > 0:
        save_session_log(
            mode=stats["mode"],
            subject=stats["subject"],
            avg_bpm=stats["avg_bpm"],
            min_bpm=stats["min_bpm"],
            max_bpm=stats["max_bpm"],
            samples=stats["buffer_count"]
        )

    monitor.stop()
    return jsonify({"success": True, "message": "Stopped"})


@app.route("/api/stats")
def api_stats():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    return jsonify(monitor.get_stats())


@app.route("/api/run_dataset_analysis", methods=["POST"])
def api_run_dataset_analysis():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    result = run_dataset_analysis("dataset", "static/plots")
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
