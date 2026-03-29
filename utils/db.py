import sqlite3
from datetime import datetime

DB_PATH = "heartrate.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password TEXT,
            created_at TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mode TEXT,
            subject TEXT,
            avg_bpm REAL,
            min_bpm REAL,
            max_bpm REAL,
            samples INTEGER,
            created_at TEXT
        )
    """)

    conn.commit()

    # default user
    cur.execute("SELECT * FROM users WHERE username = ?", ("ganesh",))
    if cur.fetchone() is None:
        cur.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            ("ganesh", "ganarm2003@gmail.com", "admin123", datetime.now().strftime("%Y-%m-%d"))
        )
        conn.commit()

    conn.close()


def register_user(username, email, password):
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password, created_at) VALUES (?, ?, ?, ?)",
            (username, email, password, datetime.now().strftime("%Y-%m-%d"))
        )
        conn.commit()
        conn.close()
        return True, "Registered"
    except sqlite3.IntegrityError:
        return False, "Username already exists"


def check_login(username, password):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user(user_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def save_session_log(mode, subject, avg_bpm, min_bpm, max_bpm, samples):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO session_logs (mode, subject, avg_bpm, min_bpm, max_bpm, samples, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        mode,
        subject,
        avg_bpm,
        min_bpm,
        max_bpm,
        samples,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()


def get_session_logs(limit=50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM session_logs ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]
