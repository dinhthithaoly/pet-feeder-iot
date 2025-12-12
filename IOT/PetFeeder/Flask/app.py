from flask import Flask, Response, render_template, jsonify, request, redirect
import cv2
from ultralytics import YOLO
import json
from datetime import datetime, timedelta

app = Flask(__name__)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("yolov8n.pt")  # model nhận cat/dog

cap = cv2.VideoCapture(0)

# -----------------------------
# GLOBAL STATUS
# -----------------------------
status = {
    "pet": "NOT FOUND",
    "pet_detected": False,
    "last_feed": None,
}

SETTINGS_FILE = "settings.json"
LOG_FILE = "feed_log.json"


# -----------------------------
# LOAD / SAVE SETTINGS
# -----------------------------
def load_settings():
    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except:
        return {
            "feed_hours": 5,
            "feed_minutes": 0,
            "portion": "MED",
        }


def save_settings(data):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


settings = load_settings()


# -----------------------------
# FEEDING LOG
# -----------------------------
def load_log():
    try:
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    except:
        return []


def save_log(data):
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_feed_log(reason):
    logs = load_log()
    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "portion": settings["portion"],
    }
    logs.append(entry)
    save_log(logs)


# -----------------------------
# PET DETECTION
# -----------------------------
def detect_pet(frame):
    results = model.predict(frame, conf=0.55, verbose=False)
    pet_found = False
    pet_label = "NOT FOUND"

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["cat", "dog"]:
                pet_found = True
                pet_label = label.upper()

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

    status["pet_detected"] = pet_found
    status["pet"] = pet_label

    return pet_found


# -----------------------------
# AUTO FEED (MODE B)
# -----------------------------
def auto_feed_if_needed():
    now = datetime.now()

    # Nếu chưa có lần cho ăn → đặt mặc định
    if status["last_feed"] is None:
        status["last_feed"] = now
        return

    interval = timedelta(hours=settings["feed_hours"], minutes=settings["feed_minutes"])

    if not status["pet_detected"]:
        return  # Không thấy pet → không cho ăn

    if now - status["last_feed"] >= interval:
        print(">>> AUTO FEED TRIGGERED!")
        status["last_feed"] = now
        add_feed_log("Pet xuất hiện & đủ thời gian")
        # (tại đây bạn gửi tín hiệu đến motor nhả thức ăn)
        return True

    return False


# -----------------------------
# STREAMING
# -----------------------------
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        pet_found = detect_pet(frame)
        auto_feed_if_needed()

        # Hiển thị trạng thái
        color = (0, 255, 0) if pet_found else (0, 0, 255)

        cv2.putText(
            frame,
            f"Pet: {status['pet']}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

        if status["last_feed"]:
            cv2.putText(
                frame,
                f"Last feed: {status['last_feed'].strftime('%H:%M:%S')}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def stream():
    return render_template("stream.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def api_status():
    return jsonify(status)


@app.route("/dashboard")
def dashboard():
    logs = load_log()
    return render_template("dashboard.html", logs=logs)


@app.route("/dashboard/clear")
def clear_dashboard():
    save_log([])
    return redirect("/dashboard")


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    global settings
    if request.method == "POST":
        settings["feed_hours"] = int(request.form.get("hours", 5))
        settings["feed_minutes"] = int(request.form.get("minutes", 0))
        settings["portion"] = request.form.get("portion", "MED")

        save_settings(settings)
        return redirect("/settings")

    return render_template("settings.html", settings=settings)


app.run(host="0.0.0.0", port=8001, debug=True)
