import os
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ==========================
# CONFIG PATHS
# ==========================
DATASET_DIR = "dataset"
TRAINER_PATH = "trainer.yml"
LABEL_MAP_CSV = "label_map.csv"
ATTENDANCE_CSV = "Attendance.csv"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)

log = logging.getLogger("attendance_app")

# ==========================
# UTILITY FUNCTIONS
# ==========================
def get_cascade():
    if not os.path.exists(CASCADE_PATH):
        st.error(f"Missing: {CASCADE_PATH}")
        st.stop()
    return cv2.CascadeClassifier(CASCADE_PATH)

def load_label_map():
    if not os.path.exists(LABEL_MAP_CSV):
        return {}
    df = pd.read_csv(LABEL_MAP_CSV)
    return dict(zip(df["ID"], df["Name"]))

def save_label_map(label_map):
    df = pd.DataFrame(list(label_map.items()), columns=["ID", "Name"])
    df.to_csv(LABEL_MAP_CSV, index=False)

def ensure_attendance():
    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=["ID", "Name", "Date", "Time"]).to_csv(
            ATTENDANCE_CSV, index=False
        )

# ==========================
# TRAIN LBPH MODEL
# ==========================
def train_model():
    cascade = get_cascade()
    face_samples = []
    ids = []
    label_map = {}

    for person_folder in os.listdir(DATASET_DIR):
        path = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(path):
            continue

        parts = person_folder.split(".")
        if len(parts) < 3:
            continue

        try:
            id_num = int(parts[1])
        except:
            continue

        name = ".".join(parts[2:])
        label_map[id_num] = name

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces = cascade.detectMultiScale(img)
            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(id_num)

    if len(face_samples) == 0:
        raise ValueError("No faces found. Capture images first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(TRAINER_PATH)
    save_label_map(label_map)

# ==========================
# STREAMLIT UI
# ==========================
st.title("AI Smart Surveillance & Attendance System")

# SIDEBAR CAPTURE
st.sidebar.header("Capture New Person")
sid = st.sidebar.number_input("ID", min_value=1, value=1)
sname = st.sidebar.text_input("Name")
snap = st.sidebar.camera_input("Take picture")

if st.sidebar.button("Save image"):
    if snap:
        folder = os.path.join(DATASET_DIR, f"User.{sid}.{sname}")
        os.makedirs(folder, exist_ok=True)
        ts = int(time.time() * 1000)
        path = f"{folder}/User.{sid}.{ts}.jpg"
        with open(path, "wb") as f:
            f.write(snap.getvalue())
        st.sidebar.success("Image saved")
    else:
        st.sidebar.warning("Capture image first")

# SIDEBAR TRAIN
st.sidebar.subheader("Train recognizer")
if st.sidebar.button("Train LBPH model"):
    with st.spinner("Training..."):
        try:
            train_model()
            st.sidebar.success("Training complete")
        except Exception as e:
            st.sidebar.error(str(e))

# ==========================
# VIDEO TRANSFORMER
# ==========================
class Transformer(VideoTransformerBase):
    def __init__(self):
        self.cascade = get_cascade()
        self.recognizer = None
        self.known_total = 0
        self.unknown_total = 0
        self.last_frame = None

        if os.path.exists(TRAINER_PATH):
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(TRAINER_PATH)
            except:
                self.recognizer = None

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                label = "Unknown"

                if self.recognizer is not None:
                    pid, conf = self.recognizer.predict(face)
                    if conf < 60:
                        label_map = load_label_map()
                        label = label_map.get(pid, f"ID {pid}")
                        self.known_total += 1
                        self._mark_attendance(pid, label)
                    else:
                        self.unknown_total += 1
                else:
                    self.unknown_total += 1

                cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(img, label, (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            cv2.putText(img,
                        f"Known:{self.known_total}  Unknown:{self.unknown_total}",
                        (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255,255,0),
                        2)

            self.last_frame = img
            return img

        except Exception:
            err = np.zeros((240,640,3), dtype=np.uint8)
            cv2.putText(err, "Stream error", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return err

    def _mark_attendance(self, pid, name):
        ensure_attendance()
        df = pd.read_csv(ATTENDANCE_CSV)
        today = datetime.now().strftime("%Y-%m-%d")
        if pid not in df[df["Date"] == today]["ID"].astype(int).tolist():
            new = {
                "ID": pid,
                "Name": name,
                "Date": today,
                "Time": datetime.now().strftime("%H:%M:%S")
            }
            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
            df.to_csv(ATTENDANCE_CSV, index=False)

# ==========================
# START CAMERA
# ==========================
st.header("Live Recognition")

webrtc_ctx = webrtc_streamer(
    key="attendance",
    video_transformer_factory=Transformer,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)

# ==========================
# LIVE COUNTS PANEL
# ==========================
st.header("Live Results")

if webrtc_ctx and webrtc_ctx.state.playing:
    transformer = webrtc_ctx.video_transformer
    if transformer:
        st.metric("Known People", transformer.known_total)
        st.metric("Unknown People", transformer.unknown_total)

        if transformer.last_frame is not None:
            rgb = cv2.cvtColor(transformer.last_frame, cv2.COLOR_BGR2RGB)
            st.image(rgb, channels="RGB")
else:
    st.info("Start camera to see results")

# ==========================
# DATABASE VIEW
# ==========================
st.header("Attendance Database")

ensure_attendance()
df = pd.read_csv(ATTENDANCE_CSV)

if st.button("Show Attendance"):
    st.dataframe(df)

st.markdown("---")
st.caption("AI Surveillance System — Streamlit Deployment")
