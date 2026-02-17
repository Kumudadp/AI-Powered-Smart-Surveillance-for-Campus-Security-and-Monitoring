import os
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO

# ==========================
# CONFIG PATHS (YOUR FILES)
# ==========================
DATASET_DIR = "dataset"
TRAINER_PATH = "trainer.yml"
LABEL_MAP_CSV = "label_map.csv"
ATTENDANCE_CSV = "Attendance.csv"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# ⭐ YOUR RENAMED CAFFE MODEL:
CAFFE_PROTO = "MobileNetSSD_deploy.prototxt"
CAFFE_MODEL = "MobileNetSSD_deploy.caffemodel"

# Create necessary folders
os.makedirs(DATASET_DIR, exist_ok=True)

# ====================================
# FUNCTIONS MUST BE DECLARED FIRST !!!
# ====================================

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
        pd.DataFrame(columns=["ID", "Name", "Date", "Time"]).to_csv(ATTENDANCE_CSV, index=False)

# ============================
# ⭐ TRAIN LBPH MODEL FUNCTION
# ============================
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
    return True

# ============================================
# STREAMLIT UI STARTS AFTER ALL FUNCTIONS EXIST
# ============================================

st.title("Face Recognition + Person & Vehicle Counting")

# -----------------------
# SIDEBAR — CAPTURE DATA
# -----------------------
st.sidebar.header("Capture New Person")
sid = st.sidebar.number_input("ID", min_value=1, value=1)
sname = st.sidebar.text_input("Name")

snap = st.sidebar.camera_input("Take picture")

if st.sidebar.button("Save image"):
    if snap:
        folder = os.path.join(DATASET_DIR, f"User.{sid}.{sname}")
        os.makedirs(folder, exist_ok=True)
        ts = int(time.time()*1000)
        path = f"{folder}/User.{sid}.{ts}.jpg"
        with open(path, "wb") as f:
            f.write(snap.getvalue())
        st.sidebar.success(f"Saved: {path}")
    else:
        st.sidebar.warning("Capture an image first.")

# ---------------------------
# SIDEBAR — TRAIN RECOGNIZER
# ---------------------------
st.sidebar.subheader("Train recognizer")
if st.sidebar.button("Train LBPH model now"):
    with st.spinner("Training..."):
        try:
            train_model()
            st.sidebar.success("Training successful!")
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

# =====================================================
# REAL-TIME RECOGNITION TRANSFORMER (YOLO + LBPH + DEDUPE)
# =====================================================

# ---- REPLACE Transformer class and webrtc start block with this code ----

import traceback
import logging
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

log = logging.getLogger("attendance_app")

class Transformer(VideoTransformerBase):
    def __init__(self):
        # Keep init lightweight and resilient so the streamer doesn't immediately die.
        try:
            self.cascade = get_cascade()
        except Exception as e:
            log.error("Cascade init failed: %s", e)
            self.cascade = None

        # LBPH recognizer: load if available, otherwise keep None
        self.recognizer = None
        try:
            if os.path.exists(TRAINER_PATH):
                # load recognizer lazily
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.recognizer.read(TRAINER_PATH)
        except Exception as e:
            log.exception("Could not load LBPH recognizer: %s", e)
            self.recognizer = None

        # YOLO: lazy load and catch errors (ultralytics can raise)
        self.yolo = None
        self.names = {}
        try:
            # Do not crash if model can't be loaded; stream will continue without YOLO
            from ultralytics import YOLO
            try:
                self.yolo = YOLO("yolov8s.pt")  # will download if needed
                self.names = self.yolo.model.names if hasattr(self.yolo, "model") else {}
            except Exception as e:
                log.exception("Failed to init YOLO model: %s", e)
                self.yolo = None
        except Exception:
            # ultralytics not installed or import fails
            self.yolo = None

        # counting/dedup state
        self.known_last = {}
        self.known_counted = set()
        self.unknown_templates = []
        self.vehicle_centers = {}
        self.KNOWN_CD = 3.0
        self.UNKNOWN_EXPIRE = 120.0
        self.UNKNOWN_COARSE = 40
        self.UNKNOWN_MATCH_THRESH = 0.65
        self.VEHICLE_CD = 3.0

        self.known_total = 0
        self.unknown_total = 0
        self.vehicle_total = 0

        # small flag to show an init error message once
        self._init_error = False

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            out = img.copy()

            detections = []
            # Try YOLO if loaded
            if self.yolo is not None:
                try:
                    results = self.yolo.predict(img, conf=0.35, imgsz=480, verbose=False)
                    if results and len(results) > 0:
                        r = results[0]
                        for b in r.boxes:
                            xy = b.xyxy[0].tolist()
                            cls_id = int(b.cls[0])
                            conf = float(b.conf[0])
                            name = self.names.get(cls_id, str(cls_id))
                            detections.append((name, conf, xy))
                except Exception as e:
                    # log but keep processing (so the stream doesn't die)
                    log.exception("YOLO predict error: %s", e)
                    # fallback to cascade face/person detection below
                    detections = []

            # Fallback to Haar-face-as-person if no YOLO detections
            if not detections:
                if self.cascade is None:
                    # nothing to detect, just show frame
                    cv2.putText(out, "No detector available", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    return out
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = self.cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
                for (x, y, fw, fh) in faces:
                    detections.append(("person", 1.0, [x, y, x+fw, y+fh]))

            # Vehicles set
            VEH = {"car", "truck", "bus", "motorcycle", "bicycle"}
            # handle vehicles
            for cls, conf, xyxy in detections:
                if cls in VEH:
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    key = (cx//self.UNKNOWN_COARSE, cy//self.UNKNOWN_COARSE)
                    now = time.time()
                    if now - self.vehicle_centers.get(key, 0) > self.VEHICLE_CD:
                        self.vehicle_total += 1
                        self.vehicle_centers[key] = now
                    cv2.rectangle(out, (x1,y1), (x2,y2), (0,165,255), 2)
                    cv2.putText(out, f"{cls} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

            # handle persons
            for cls, conf, xyxy in detections:
                if cls != "person":
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w-1, x2), min(h-1, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                person_crop = img[y1:y2, x1:x2]
                if person_crop.size == 0:
                    continue
                gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                faces_in_crop = self.cascade.detectMultiScale(gray_crop, scaleFactor=1.1, minNeighbors=4)

                if len(faces_in_crop) == 0:
                    # spatial dedupe
                    cx, cy = (x1 + x2)//2, (y1 + y2)//2
                    self._dedupe_unknown(cx, cy)
                    cv2.rectangle(out, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(out, "Unknown person", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    continue

                for (fx, fy, fw, fh) in faces_in_crop:
                    absx, absy = x1 + fx, y1 + fy
                    if absx < 0 or absy < 0 or absx+fw > w or absy+fh > h:
                        continue
                    face = img[absy:absy+fh, absx:absx+fw]
                    if face.size == 0:
                        continue
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    try:
                        face_gray = cv2.resize(face_gray, (200,200), interpolation=cv2.INTER_AREA)
                    except Exception:
                        pass
                    face_gray = cv2.equalizeHist(face_gray)

                    recognized = False
                    pid = None
                    conf_score = 999
                    if self.recognizer is not None:
                        try:
                            pid, conf_score = self.recognizer.predict(face_gray)
                            if conf_score < 60:
                                recognized = True
                        except Exception as e:
                            log.exception("LBPH predict error: %s", e)

                    if recognized:
                        label_map = load_label_map()
                        name = label_map.get(pid, f"ID_{pid}")
                        now = time.time()
                        if now - self.known_last.get(pid, 0) > self.KNOWN_CD:
                            if pid not in self.known_counted:
                                self.known_total += 1
                                self.known_counted.add(pid)
                            self.known_last[pid] = now

                        ensure_attendance()
                        df = pd.read_csv(ATTENDANCE_CSV)
                        today = datetime.now().strftime("%Y-%m-%d")
                        if pid not in df[df["Date"] == today]["ID"].astype(int).tolist():
                            new = {"ID": pid, "Name": name, "Date": today, "Time": datetime.now().strftime("%H:%M:%S")}
                            df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
                            df.to_csv(ATTENDANCE_CSV, index=False)

                        cv2.rectangle(out, (absx, absy), (absx+fw, absy+fh), (0,255,0), 2)
                        cv2.putText(out, name, (absx, absy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    else:
                        cx = absx + fw//2
                        cy = absy + fh//2
                        self._dedupe_unknown(cx, cy)
                        cv2.rectangle(out, (absx, absy), (absx+fw, absy+fh), (0,0,255), 2)
                        cv2.putText(out, "Unknown", (absx, absy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

            # overlay counts
            cv2.putText(out, f"Known:{self.known_total}  Unknown:{self.unknown_total}  Vehicles:{self.vehicle_total}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            return out

        except Exception as e:
            # catch ANY error and show message on frame instead of stopping stream
            tb = traceback.format_exc()
            log.error("Transformer.recv error: %s\n%s", e, tb)
            err_img = np.zeros((240,640,3), dtype=np.uint8)
            cv2.putText(err_img, "Stream error, check logs", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            # also write the traceback to a file to help debugging
            with open("webrtc_error.log", "a") as f:
                f.write(f"{datetime.now().isoformat()} - recv error:\n{tb}\n\n")
            return err_img

    def _dedupe_unknown(self, cx, cy):
        now = time.time()
        for t in self.unknown_templates:
            if abs(t["cx"] - cx) < self.UNKNOWN_COARSE and abs(t["cy"] - cy) < self.UNKNOWN_COARSE:
                if now - t["last"] < self.UNKNOWN_EXPIRE:
                    t["last"] = now
                    return
        self.unknown_templates.append({"cx": cx, "cy": cy, "last": now})
        self.unknown_total += 1


# Start camera stream with async_transform True and return context so user can see state
st.header("Live Recognition")
webrtc_ctx = webrtc_streamer(
    key="attendance_live",
    video_transformer_factory=Transformer,
    rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    video_html_attrs={"controls": False, "autoPlay": True}
)

# Show basic state so you know if it's running
if webrtc_ctx and webrtc_ctx.state.playing:
    st.success("Camera stream running. Allow camera permission in your browser and check logs for detections.")
else:
    st.info("Click Start/Allow camera. If the stream stops, check 'webrtc_error.log' for details.")
# ---- Add this AFTER your webrtc_streamer(...) call in app.py ----
import time
from pathlib import Path

st.header("Live Recognition Controls & Results")

# If webrtc_ctx was created earlier by webrtc_streamer(...)
# (ensure the key matches: e.g., key="attendance_live")
try:
    webrtc_ctx  # check if webrtc_ctx exists
except NameError:
    webrtc_ctx = None

if webrtc_ctx is None:
    st.info("Start the camera stream above to see live results.")
else:
    # placeholders for live values
    counts_ph = st.empty()
    latest_frame_ph = st.empty()
    summary_ph = st.empty()

    # stop button
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Show Results (refresh)"):
            st.experimental_rerun()
    with col2:
        if st.button("Stop Stream"):
            # stop the stream gracefully
            try:
                webrtc_ctx.stop()
                st.success("Stream stopped.")
            except Exception as e:
                st.error(f"Could not stop stream: {e}")

    # Poll transformer state and update placeholders while stream is running
    # This loop will run only on user interaction in Streamlit (avoid infinite tight loop)
    if webrtc_ctx.state.playing:
        st.success("Camera stream running. Fetching live counts...")
        # try to access transformer object
        t = None
        try:
            t = webrtc_ctx.video_transformer
        except Exception:
            t = None

        # If transformer exists, show its counters
        if t is not None:
            # update UI a few times to let user see values (non-blocking)
            for _ in range(30):  # update ~30 times (~30 * 0.5s = 15s)
                try:
                    known = getattr(t, "known_total", 0)
                    unknown = getattr(t, "unknown_total", 0)
                    vehicles = getattr(t, "vehicle_total", 0)
                except Exception:
                    known = unknown = vehicles = 0

                counts_ph.markdown(f"**Live counts** — Known: **{known}**  |  Unknown: **{unknown}**  |  Vehicles: **{vehicles}**")
                # show small snapshot if available (transformer may expose last_frame as ndarray)
                frame_img = None
                try:
                    last_frame = getattr(t, "last_frame", None)
                    if last_frame is not None:
                        # last_frame expected as BGR ndarray; convert to RGB for Streamlit
                        import cv2
                        import numpy as np
                        rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                        frame_img = rgb
                except Exception:
                    frame_img = None

                if frame_img is not None:
                    latest_frame_ph.image(frame_img, caption="Live frame (snapshot)", channels="RGB", use_column_width=True)
                else:
                    latest_frame_ph.info("Live video preview not available (browser shows video).")

                time.sleep(0.5)  # pause so updates are visible

            # After polling, show final session summary
            summary_ph.markdown(
                f"### Session summary\n\n"
                f"- Known people detected: **{known}**\n\n"
                f"- Unknown people detected: **{unknown}**\n\n"
                f"- Vehicles detected: **{vehicles}**\n\n"
                f"Attendance file: `{ATTENDANCE_CSV}`"
            )
        else:
            counts_ph.info("Transformer not ready yet — wait a second and press 'Show Results' again.")
    else:
        st.info("Stream is not running. Click Start Camera to begin.")

# Show the local model file path (your renamed caffemodel)
st.markdown("---")
st.markdown("**Local MobileNet Caffe model (renamed):**")
local_caffe_path = Path("MobileNetSSD_deploy.caffemodel").resolve()
st.code(str(local_caffe_path))

# ----------------------------
# ---------------------------
# Show Database button & summary
# ---------------------------
import json
from pathlib import Path

st.header("Database of Recognized People & Session Counts")

# path to your attendance CSV
ensure_attendance()  # make sure the file exists
att_path = Path(ATTENDANCE_CSV)

# local model path (developer-provided / renamed file) - delivered as requested
MODEL_LOCAL_PATH = "/mnt/data/mobilenet_iter_73000.caffemodel"
st.markdown(f"**Local MobileNet Caffe model path:** `{MODEL_LOCAL_PATH}`")

if st.button("Show Database"):
    # Read attendance CSV and display (includes timestamp columns Date, Time)
    try:
        df_att = pd.read_csv(att_path)
        # Ensure Date/Time columns exist; show exactly as in your screenshot
        if "Date" in df_att.columns and "Time" in df_att.columns:
            st.subheader("Attendance records (with timestamp)")
            st.dataframe(df_att.astype(str))
        else:
            st.warning("Attendance file missing Date/Time columns; showing raw file.")
            st.dataframe(df_att.astype(str))
    except Exception as e:
        st.error(f"Could not read attendance file: {e}")
        df_att = pd.DataFrame(columns=["ID","Name","Date","Time"])

    # Aggregate counts from attendance file
    total_rows = len(df_att)
    unique_persons = int(df_att["ID"].nunique()) if "ID" in df_att.columns and not df_att.empty else 0
    st.markdown(f"- **Total attendance rows**: {total_rows}  \n- **Unique persons in file**: {unique_persons}")

    # Try to read live counters from the running transformer first
    counts = {"known": 0, "unknown": 0, "vehicles": 0, "source": "none"}
    transformer = None
    try:
        transformer = getattr(webrtc_ctx, "video_transformer", None)
    except Exception:
        transformer = None

    if transformer is not None:
        # read from transformer attributes (live)
        counts["known"] = int(getattr(transformer, "known_total", 0))
        counts["unknown"] = int(getattr(transformer, "unknown_total", 0))
        counts["vehicles"] = int(getattr(transformer, "vehicle_total", 0))
        counts["source"] = "live_transformer"
    else:
        # fallback: read the session_counters.json persisted by transformer
        try:
            with open("session_counters.json", "r") as jf:
                jc = json.load(jf)
                counts["known"] = int(jc.get("known", 0))
                counts["unknown"] = int(jc.get("unknown", 0))
                counts["vehicles"] = int(jc.get("vehicles", 0))
                counts["source"] = "session_counters.json"
        except Exception:
            counts["source"] = "none"

    st.subheader("Session counts")
    st.markdown(f"- **Known people (counted this session):** {counts['known']}  \n- **Unknown people (counted this session):** {counts['unknown']}  \n- **Vehicles (counted this session):** {counts['vehicles']}")
    st.caption(f"Counts source: {counts['source']}")

    # Option: show unique persons with last timestamp from Attendance file
    if not df_att.empty and "ID" in df_att.columns:
        st.subheader("Unique persons and last seen time")
        try:
            grouped = df_att.sort_values(["ID","Date","Time"]).groupby("ID").tail(1)[["ID","Name","Date","Time"]]
            st.table(grouped.reset_index(drop=True))
        except Exception:
            st.info("Could not compute per-person last seen times — raw table shown above.")
