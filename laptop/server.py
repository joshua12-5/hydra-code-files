import cv2, time, os, numpy as np
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO
from ultralytics import YOLO
from threading import Lock

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

os.makedirs("recordings", exist_ok=True)

WIDTH, HEIGHT = 640, 480
model = YOLO("yolo11n.pt")
CLASSES = [0, 24, 26, 28]  # person + clothing related

raw_frame = None
cv_frame = None
lock = Lock()

recording = False
raw_writer = cv_writer = None
gps_logs = []

# ---------- VIDEO FROM PI ----------
@socketio.on("video")
def receive_video(data):
    global raw_frame, cv_frame
    jpg = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (WIDTH, HEIGHT))

    annotated = frame.copy()
    results = model.predict(frame, classes=CLASSES, conf=0.25, verbose=False)

    if results and results[0].boxes:
        for b in results[0].boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0])
            label = model.names[int(b.cls[0])]
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,255,0),2)
            cv2.putText(annotated,label,(x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),2)

    with lock:
        raw_frame = frame
        cv_frame = annotated
        if recording:
            raw_writer.write(raw_frame)
            cv_writer.write(cv_frame)

# ---------- GPS ----------
@socketio.on("gps")
def gps_log(data):
    gps_logs.append(data)

@socketio.on("manual_gps")
def manual_gps():
    gps_logs.append({
        "time": time.time(),
        "lat": "N/A",
        "lon": "N/A",
        "manual": True
    })

# ---------- AUDIO ----------
@socketio.on("pi_audio")
def forward_audio(data):
    socketio.emit("pi_audio", data)

@socketio.on("laptop_audio")
def send_audio(data):
    socketio.emit("laptop_audio", data)

# ---------- RECORD ----------
@socketio.on("toggle_record")
def toggle_record(active):
    global recording, raw_writer, cv_writer
    recording = active

    if active:
        ts = time.strftime("%Y%m%d_%H%M%S")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        raw_writer = cv2.VideoWriter(f"recordings/raw_{ts}.mp4", fourcc, 20, (WIDTH, HEIGHT))
        cv_writer = cv2.VideoWriter(f"recordings/cv_{ts}.mp4", fourcc, 20, (WIDTH, HEIGHT))
    else:
        if raw_writer:
            raw_writer.release()
            cv_writer.release()

# ---------- STREAMS ----------
def gen(get):
    while True:
        with lock:
            frame = get()
            if frame is None:
                continue
            _, jpg = cv2.imencode(".jpg", frame)
        yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n"+jpg.tobytes()+b"\r\n"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/raw")
def raw():
    return Response(gen(lambda: raw_frame), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/cv")
def cv():
    return Response(gen(lambda: cv_frame), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/logs")
def logs():
    return jsonify(gps_logs[-100:])

socketio.run(app, host="0.0.0.0", port=5000)

#---------- Pip installation Instructions ----------
# To install required packages, run the following commands:
# pip install opencv-python flask flask-socketio pyserial pynmea2 sounddevice soundfile numpy
