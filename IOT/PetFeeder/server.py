from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
import os
from datetime import datetime
from ultralytics import YOLO
import random

app = Flask(__name__)
LOG_FILE = "motion_log.txt"

# ===============================
# YOLO MODEL (COCO)
# ===============================
PET_MODEL = YOLO("yolov8n.pt")
PET_THRESHOLD = 0.25

# ===============================
# GLOBAL STATES
# ===============================
last_pir = 0
last_rfid = None
pet_detected_flag = False
behavior_score = 0

camera = None
latest_frame = None
last_gray = None
camera_lock = threading.Lock()

# Cooldown tránh spam log
last_no_pet_log = 0
NO_PET_COOLDOWN = 5  # giây


# ===============================
# LOGGING
# ===============================
def write_log(msg):
    entry = f"{datetime.now().strftime('%H:%M:%S')} - {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")
    print("📝", entry)


def log_motion():
    write_log("PIR: phát hiện chuyển động")


def log_yolo(label, conf):
    write_log(f"AI xác minh PET: {label} ({conf:.2f})")


def log_no_pet():
    write_log("AI xác minh: Không phát hiện thú cưng")


def log_rfid(tag):
    write_log(f"RFID: Mèo của Vân mang thẻ {tag}")


# ===============================
# CAMERA INIT
# ===============================
def init_camera():
    global camera
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (0, "Default")
    ]

    for backend, name in backends:
        try:
            cam = cv2.VideoCapture(0, backend) if backend != 0 else cv2.VideoCapture(0)
            if cam.isOpened():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera = cam
                print(f"📷 Camera started using {name}")
                return
        except:
            pass

    print("❌ Camera not found!")


# ===============================
# CAMERA LOOP (YOLO only when PIR=1)
# ===============================
def camera_loop():
    global latest_frame, last_gray, camera
    global pet_detected_flag, behavior_score, last_pir
    global last_no_pet_log

    while True:
        if camera is None or not camera.isOpened():
            init_camera()
            time.sleep(1)
            continue

        ret, frame = camera.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ------------ MOTION DETECTION ------------ #
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_detected = False

        if last_gray is not None:
            diff = cv2.absdiff(last_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) > 800:
                    motion_detected = True
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        last_gray = gray

        if motion_detected:
            behavior_score = min(100, behavior_score + 3)

        # ------------ YOLO DETECTION (ONLY IF PIR=1) ------------ #
        pet_label = "Khong thay"
        pet_conf = 0.0

        # Chỉ reset trạng thái khi PIR = 0
        if last_pir == 0:
            pet_detected_flag = False

        if last_pir == 1:

            results = PET_MODEL.predict(frame, conf=PET_THRESHOLD, verbose=False)
            found_pet = False

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = PET_MODEL.names[cls]

                    if label in ["dog", "cat"] and conf >= PET_THRESHOLD:
                        found_pet = True
                        pet_detected_flag = True
                        pet_label = label
                        pet_conf = conf
                        log_yolo(label, conf)

                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(frame, f"{label} {conf:.2f}",
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 255), 2)

            
            if not found_pet:
                now = time.time()
                if now - last_no_pet_log >= NO_PET_COOLDOWN:
                    log_no_pet()
                    last_no_pet_log = now

        # ------------ TEXT OVERLAY ------------ #
        cv2.putText(
            frame,
            f"Pet: {pet_label} ({pet_conf:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 200, 0), 2
        )

        # ------------ UPDATE STREAM ------------ #
        with camera_lock:
            latest_frame = frame.copy()

        time.sleep(0.03)


# ===============================
# ARDUINO SIMULATION LOOP
# ===============================
def arduino_simulation_loop():
    global last_pir, last_rfid, behavior_score

    while True:
        action = random.choice(["pir", "rfid", "none"])

        if action == "pir":
            last_pir = random.choice([0, 1])
            if last_pir == 1:
                log_motion()
                behavior_score = min(100, behavior_score + 5)

        elif action == "rfid":
            last_rfid = "VAN_CAT_001"
            log_rfid(last_rfid)
            behavior_score = min(100, behavior_score + 3)

        time.sleep(3)


# ===============================
# DAILY RESET LOG AT 00:00
# ===============================
def daily_log_reset():
    last_day = datetime.now().day

    while True:
        now = datetime.now()
        if now.day != last_day:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write("")
            print("🗑️ Log reset for new day:", now.strftime("%Y-%m-%d"))
            last_day = now.day

        time.sleep(60)


# ===============================
# STREAM VIDEO
# ===============================
def gen_frames():
    global latest_frame
    while True:
        with camera_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ===============================
# DASHBOARD API
# ===============================
@app.route("/sensor_status")
def sensor_status():
    return jsonify({
        "pir": last_pir,
        "rfid": last_rfid,
        "pet_detected": pet_detected_flag,
        "behavior_score": behavior_score
    })


@app.route("/camera_status")
def camera_status():
    return jsonify({"active": True})


@app.route("/motion_stats")
def motion_stats():
    if not os.path.exists(LOG_FILE):
        return jsonify([])

    stats = {}
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            minute = line.split(" - ")[0][:5]
            stats[minute] = stats.get(minute, 0) + 1

    return jsonify([
        {"time": k, "count": v}
        for k, v in sorted(stats.items())
    ])


@app.route("/get_logs")
def get_logs():
    if not os.path.exists(LOG_FILE):
        return jsonify([])

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return jsonify([line.strip() for line in f])


@app.route("/")
def index():
    return render_template("index.html")


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("🚀 SYSTEM MODE D — Mèo của Vân + Daily Reset + Stable Detection")
    init_camera()

    threading.Thread(target=camera_loop, daemon=True).start()
    threading.Thread(target=arduino_simulation_loop, daemon=True).start()
    threading.Thread(target=daily_log_reset, daemon=True).start()

    app.run(debug=True, threaded=True, use_reloader=False)
