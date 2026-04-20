from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response, flash
import os
from utils.db import init_db, register_user, check_login, get_user, save_session_log, get_session_logs
from utils.monitor import monitor
from utils.dataset_analysis import run_dataset_analysis, load_latest_dataset_summary

app = Flask(__name__)
app.secret_key = "rppg-secret-key"

# create important folders if not present
os.makedirs("static/plots", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# initialize database
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

    return render_template(
        "realtime.html",
        user=user,
        subjects=subjects
    )


@app.route("/dataset-analysis")
def dataset_analysis():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    summary = load_latest_dataset_summary()

    return render_template(
        "dataset_analysis.html",
        user=user,
        summary=summary
    )


@app.route("/model-info")
def model_info():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    summary = load_latest_dataset_summary()

    return render_template(
        "model_info.html",
        user=user,
        summary=summary
    )


@app.route("/history")
def history():
    if not login_required():
        return redirect(url_for("login"))

    user = get_user(session["user_id"])
    logs = get_session_logs(limit=100)

    return render_template(
        "history.html",
        user=user,
        logs=logs
    )


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
    return jsonify({
        "success": ok,
        "message": msg
    })


@app.route("/api/start_demo", methods=["POST"])
def api_start_demo():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    data = request.get_json(silent=True) or {}
    subject = data.get("subject", "")
    use_dl = bool(data.get("use_dl", False))

    ok, msg = monitor.start_demo("dataset", subject, use_dl=use_dl)

    return jsonify({
        "success": ok,
        "message": msg
    })


@app.route("/api/stop_monitor", methods=["POST"])
def api_stop_monitor():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    stats = monitor.get_stats()

    if stats.get("avg_bpm", 0) > 0:
        save_session_log(
            mode=stats.get("mode", "unknown"),
            subject=stats.get("subject", ""),
            avg_bpm=stats.get("avg_bpm", 0),
            min_bpm=stats.get("min_bpm", 0),
            max_bpm=stats.get("max_bpm", 0),
            samples=stats.get("buffer_count", 0)
        )

    monitor.stop()

    return jsonify({
        "success": True,
        "message": "Monitoring stopped successfully"
    })


@app.route("/api/stats")
def api_stats():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    stats = monitor.get_stats()

    # status field will come from monitor.py after we update it there
    return jsonify(stats)


@app.route("/api/run_dataset_analysis", methods=["POST"])
def api_run_dataset_analysis():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    # try loading already saved summary first
    cached_summary = load_latest_dataset_summary()

    if cached_summary:
        return jsonify({
            "success": True,
            "message": "Loaded saved dataset analysis results",
            "cached": True,
            "data": cached_summary
        })

    # if no cached result found, run fresh analysis
    result = run_dataset_analysis("dataset", "static/plots")

    if isinstance(result, dict):
        result["cached"] = False
        return jsonify(result)

    return jsonify({
        "success": False,
        "message": "Dataset analysis failed"
    })


@app.route("/api/refresh_dataset_analysis", methods=["POST"])
def api_refresh_dataset_analysis():
    if not login_required():
        return jsonify({"success": False, "message": "Unauthorized"}), 401

    # force fresh run even if cached summary exists
    result = run_dataset_analysis("dataset", "static/plots")

    if isinstance(result, dict):
        result["cached"] = False
        return jsonify(result)

    return jsonify({
        "success": False,
        "message": "Dataset analysis refresh failed"
    })


if __name__ == "__main__":
    app.run(debug=True)
