import cv2, time, socketio, serial, pynmea2, sounddevice as sd, soundfile as sf
import numpy as np
from threading import Thread

SERVER_IP = "LAPTOP_IP_HERE"
sio = socketio.Client()

# ---------- GPS ----------
gps = serial.Serial("/dev/serial0", baudrate=9600, timeout=1)

def read_gps():
    while True:
        try:
            line = gps.readline().decode(errors="ignore")
            if line.startswith("$GPGGA") or line.startswith("$GPRMC"):
                msg = pynmea2.parse(line)
                if msg.latitude and msg.longitude:
                    sio.emit("gps", {
                        "time": time.time(),
                        "lat": msg.latitude,
                        "lon": msg.longitude,
                        "manual": False
                    })
        except:
            pass

# ---------- VIDEO ----------
cap = cv2.VideoCapture(0)

def video_loop():
    while True:
        ret, frame = cap.read()
        if ret:
            _, jpg = cv2.imencode(".jpg", frame)
            sio.emit("video", jpg.tobytes())
        time.sleep(0.03)

# ---------- AUDIO MIC ----------
def mic_loop():
    def callback(indata, frames, time_info, status):
        sio.emit("pi_audio", indata.copy().tobytes())

    with sd.InputStream(channels=1, samplerate=16000, callback=callback):
        while True:
            time.sleep(1)

# ---------- AUDIO PLAYBACK ----------
@sio.on("laptop_audio")
def play_audio(data):
    audio = np.frombuffer(data, dtype=np.int16)
    sd.play(audio, samplerate=16000)

# ---------- START ----------
sio.connect(f"http://{SERVER_IP}:5000")
Thread(target=video_loop, daemon=True).start()
Thread(target=read_gps, daemon=True).start()
Thread(target=mic_loop, daemon=True).start()

while True:
    time.sleep(1)



#---------- Pipe Installation Instructions ----------
# To install required packages, run the following commands:
# pip install opencv-python flask flask-socketio pyserial pynmea2 sounddevice soundfile numpy 

