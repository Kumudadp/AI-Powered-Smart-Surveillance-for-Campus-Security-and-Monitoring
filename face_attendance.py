"""
face_attendance.py
Single-file simplified face-recognition attendance system.

Features:
1. Capture images for a person (dataset/)
2. Train LBPH recognizer (trainer.yml)
3. Recognize people in real-time and mark attendance (Attendance.csv)
4. Show attendance

Requirements:
- Python 3.8+
- pip install opencv-contrib-python numpy pandas

Place `haarcascade_frontalface_default.xml` in the same directory as this file.

Usage:
- Run the script and follow the menu:
    python face_attendance.py

Notes:
- The trainer uses OpenCV's LBPHFaceRecognizer (from opencv-contrib).
- Attendance is recorded once per person per day (no duplicate entries on same day).

"""

import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
import sys

# Paths and config
DATASET_DIR = "dataset"
TRAINER_PATH = "trainer.yml"
CASCADE_PATH = "haarcascade_frontalface_default.xml"
ATTENDANCE_CSV = "Attendance.csv"
NUM_IMAGES = 60  # images to capture per person

# Ensure folders exist
os.makedirs(DATASET_DIR, exist_ok=True)

# Helper: load cascade
def get_cascade():
    if not os.path.exists(CASCADE_PATH):
        print(f"ERROR: Haar cascade not found at '{CASCADE_PATH}'. Please place the XML file here.")
        sys.exit(1)
    return cv2.CascadeClassifier(CASCADE_PATH)

# 1) Capture images
def get_next_id():
    """Return next available numeric ID by scanning existing dataset folders."""
    max_id = 0
    for person_folder in os.listdir(DATASET_DIR):
        parts = person_folder.split('.')
        if len(parts) >= 2:
            try:
                pid = int(parts[1])
                if pid > max_id:
                    max_id = pid
            except Exception:
                continue
    return max_id + 1

def capture_images():
    cascade = get_cascade()
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open webcam")
        return

    # Some environments (like certain IDEs or headless containers) may not support input().
    # Catch OSError and fallback to automatic ID/name generation.
    try:
        id_input = input("Enter numeric ID for the person (e.g. 1): ").strip()
        name = input("Enter name for the person: ").strip()
    except OSError:
        print("Warning: stdin not available in this environment. Using automatic ID and name.")
        id_input = None
        name = None

    if id_input is None or not id_input.isdigit():
        id_num = get_next_id()
        if not name:
            name = f"User{int(datetime.now().timestamp())}"
        print(f"Using ID {id_num} and name '{name}'")
    else:
        id_num = int(id_input)

    person_dir = os.path.join(DATASET_DIR, f"User.{id_num}.{name}")
    os.makedirs(person_dir, exist_ok=True)

    print("Look at camera. Capturing images... (press 'q' to stop early)")
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f"User.{id_num}.{count}.jpg")
            try:
                cv2.imwrite(img_path, face_img)
            except Exception as e:
                print(f"Warning: could not write image to disk: {e}")
                cam.release()
                cv2.destroyAllWindows()
                return
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{name} - {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow('Capturing Images - Press q to stop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= NUM_IMAGES:
            print(f"Captured {count} images for {name} (ID {id_num})")
            break

    cam.release()
    cv2.destroyAllWindows()

# 2) Train model
def train_model():
    # Prepare data
    face_samples = []
    ids = []
    label_map = {}  # id -> name

    cascade = get_cascade()

    # Walk dataset
    for person_folder in os.listdir(DATASET_DIR):
        folder_path = os.path.join(DATASET_DIR, person_folder)
        if not os.path.isdir(folder_path):
            continue
        # Expect folder name like User.<id>.<name>
        parts = person_folder.split('.')
        if len(parts) < 3:
            print(f"Skipping unexpected folder: {person_folder}")
            continue
        try:
            id_num = int(parts[1])
        except ValueError:
            print(f"Skipping folder with non-numeric ID: {person_folder}")
            continue
        name = '.'.join(parts[2:])
        label_map[id_num] = name

        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces = cascade.detectMultiScale(img)
            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(id_num)

    if len(face_samples) == 0:
        print("No training images found. Use the capture option first.")
        return

    # Create LBPH face recognizer
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception as e:
        print("Error: OpenCV-contrib is required (cv2.face). Install with: pip install opencv-contrib-python")
        raise

    recognizer.train(face_samples, np.array(ids))
    recognizer.write(TRAINER_PATH)

    # Save simple label map to CSV for lookup
    label_df = pd.DataFrame(list(label_map.items()), columns=['ID','Name'])
    label_df.to_csv('label_map.csv', index=False)

    print(f"Model trained and saved to '{TRAINER_PATH}'. Labels saved to 'label_map.csv'.")

# 3) Recognize and mark attendance
# --- REPLACE your previous mark_attendance() WITH THIS FUNCTION ---

# --- Replace your mark_attendance() with this function ---
def mark_attendance():
    """
    YOLO (or other detector) + LBPH-based attendance with robust unknown-person deduplication.
    - Uses histogram matching to avoid counting the same unknown person repeatedly.
    - Keeps existing behavior for known people and attendance (one mark per day).
    """

    import time

    # Optional: path to your MobileNet model (if you still want to use it as fallback)
    # uploaded model path (already available in your workspace)
    MOBILE_NET_PATH = "/mnt/data/mobilenet_iter_73000.caffemodel"

    # Load LBPH recognizer (attendance identity)
    if not os.path.exists(TRAINER_PATH):
        print(f"Trainer file '{TRAINER_PATH}' not found. Train the model first.")
        return
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_PATH)
    except Exception as e:
        print("Error loading trainer. Ensure opencv-contrib is installed and trainer.yml exists.")
        raise

    # load labels for faces
    label_map = {}
    if os.path.exists('label_map.csv'):
        df_lbl = pd.read_csv('label_map.csv')
        for _, row in df_lbl.iterrows():
            label_map[int(row['ID'])] = row['Name']

    # Prepare attendance file
    if not os.path.exists(ATTENDANCE_CSV):
        pd.DataFrame(columns=['ID','Name','Date','Time']).to_csv(ATTENDANCE_CSV, index=False)
    attendance_df = pd.read_csv(ATTENDANCE_CSV)
    today_str = datetime.now().strftime('%Y-%m-%d')
    marked_today = set(attendance_df[attendance_df['Date'] == today_str]['ID'].astype(int).tolist()) if not attendance_df.empty else set()

    # Load detector: prefer YOLOv8 if available, otherwise rely on Haar cascade + MobileNet SSD fallback
    use_yolo = False
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8s.pt")  # will auto-download if needed
        use_yolo = True
        print("YOLOv8 loaded for object detection.")
    except Exception:
        yolo = None
        print("YOLO not available — falling back to Haar cascade person detection (less accurate).")

    cascade = get_cascade()

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open webcam")
        return

    # Counting & cooldown state
    known_counted = set()
    known_last_seen = {}     # id -> last timestamp when counted
    unknown_templates = []   # list of dicts: {'hist': hist_array, 'first_seen':ts, 'last_seen':ts}
    vehicle_centers = {}     # center_key -> last_timestamp

    # Parameters (tweakable)
    KNOWN_CD = 3.0               # seconds: don't recount same known id within this
    VEHICLE_CD = 3.0
    UNKNOWN_MATCH_THRESH = 0.65  # histogram correlation threshold -> treat as same unknown
    UNKNOWN_EXPIRE_SEC = 120.0   # remove unknown templates older than this (seconds)
    UNKNOWN_COARSE = 40          # coarse center grid to reduce duplicates
    HIST_SIZE = (64,)            # histogram bins for grayscale
    HIST_RANGE = [0,256]

    known_total = 0
    unknown_total = 0
    vehicle_total = 0

    print("Starting recognition with improved unknown deduplication. Press 'q' to quit.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        frame_display = frame.copy()

        # === 1) Object detection: use YOLO if available to find persons & vehicles ===
        detections = []  # list of tuples (cls_name, conf, [x1,y1,x2,y2])
        if use_yolo:
            results = yolo.predict(frame, conf=0.35, imgsz=640, verbose=False)
            if len(results) > 0:
                res = results[0]
                for box in res.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = yolo.model.names[cls_id] if hasattr(yolo.model, "names") else str(cls_id)
                    detections.append((cls_name, conf, xyxy))
        else:
            # fallback: detect faces / persons via Haar cascade (less accurate); treat every face as a person bbox
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            for (x,y,wf,hf) in faces:
                detections.append(("person", 1.0, [x, y, x+wf, y+hf]))

        # === 2) Vehicle counting (YOLO detections of vehicle classes) ===
        VEHICLE_SET = {"car","truck","bus","motorcycle","bicycle"}
        for cls_name, conf, xyxy in detections:
            if cls_name in VEHICLE_SET:
                x1, y1, x2, y2 = map(int, xyxy)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                key = (cx // UNKNOWN_COARSE, cy // UNKNOWN_COARSE)
                now_t = time.time()
                last = vehicle_centers.get(key, 0)
                if now_t - last > VEHICLE_CD:
                    vehicle_total += 1
                    vehicle_centers[key] = now_t
                    print(f"Vehicle counted ({cls_name}) total: {vehicle_total}")
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0,165,255), 2)
                cv2.putText(frame_display, f"{cls_name} {conf:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        # === 3) Person handling: for each person, attempt face detection then LBPH recognition ===
        for cls_name, conf, xyxy in detections:
            if cls_name != "person":
                continue
            x1, y1, x2, y2 = map(int, xyxy)
            # Clip coords
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            # try to find a face inside person bbox (better than whole person)
            faces_in_crop = cascade.detectMultiScale(gray_crop, scaleFactor=1.1, minNeighbors=4)
            if len(faces_in_crop) == 0:
                # no face found; treat as unknown person presence (but deduplicate spatially)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                key = (cx // UNKNOWN_COARSE, cy // UNKNOWN_COARSE)
                now_t = time.time()
                # Use simple spatial cooldown: if we've seen this coarse center recently, skip
                last = next((t for (k,t) in vehicle_centers.items() if k==key), 0)
                # Actually reuse vehicle_centers dict? avoid collision: use unknown spatial dict
                # But to keep simple, just skip increment if a template exists near center recently
                matched_recent = False
                for templ in unknown_templates:
                    # If a template's last_seen is recent and center is near (approx via bounding box), treat as same
                    if now_t - templ['last_seen'] < UNKNOWN_EXPIRE_SEC and abs(templ.get('center_x', cx)-cx) < UNKNOWN_COARSE and abs(templ.get('center_y', cy)-cy) < UNKNOWN_COARSE:
                        matched_recent = True
                        templ['last_seen'] = now_t
                        break
                if not matched_recent:
                    # Add a lightweight template entry (no hist) to mark this spatially
                    unknown_templates.append({'hist': None, 'first_seen': now_t, 'last_seen': now_t, 'center_x': cx, 'center_y': cy})
                    unknown_total += 1
                    print(f"Person presence counted as unknown (no face): total unknown {unknown_total}")
                cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame_display, "Unknown person", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                continue

            # There can be multiple faces within; handle each face detected in this person bbox
            for (fx, fy, fw, fh) in faces_in_crop:
                # Convert face coordinates to frame coordinates
                abs_x = x1 + fx
                abs_y = y1 + fy
                abs_w = fw
                abs_h = fh
                # crop & normalize face patch
                face_patch = frame[abs_y:abs_y+abs_h, abs_x:abs_x+abs_w]
                if face_patch.size == 0:
                    continue
                face_gray = cv2.cvtColor(face_patch, cv2.COLOR_BGR2GRAY)
                face_gray = cv2.resize(face_gray, (200, 200), interpolation=cv2.INTER_AREA)
                # equalize to reduce lighting differences
                face_gray = cv2.equalizeHist(face_gray)

                # Try LBPH prediction (identity)
                recognized = False
                try:
                    id_pred, confidence = recognizer.predict(face_gray)
                    if confidence < 60:
                        recognized = True
                except Exception:
                    recognized = False

                cx = abs_x + abs_w // 2
                cy = abs_y + abs_h // 2
                center_key = (cx // UNKNOWN_COARSE, cy // UNKNOWN_COARSE)
                now_t = time.time()

                if recognized:
                    name = label_map.get(id_pred, f"ID_{id_pred}")
                    last_seen = known_last_seen.get(id_pred, 0)
                    if now_t - last_seen > KNOWN_CD:
                        if id_pred not in known_counted:
                            known_total += 1
                            known_counted.add(id_pred)
                            print(f"Known person counted: {name} (total known: {known_total})")
                        known_last_seen[id_pred] = now_t

                    # mark attendance once per day
                    if id_pred not in marked_today:
                        now_dt = datetime.now()
                        new_row = {'ID': id_pred, 'Name': name, 'Date': now_dt.strftime('%Y-%m-%d'), 'Time': now_dt.strftime('%H:%M:%S')}
                        attendance_df = pd.concat([attendance_df, pd.DataFrame([new_row])], ignore_index=True)
                        attendance_df.to_csv(ATTENDANCE_CSV, index=False)
                        marked_today.add(id_pred)
                        print(f"Marked attendance for {name} (ID {id_pred}) at {new_row['Time']}")

                    # draw bounding box & name
                    cv2.rectangle(frame_display, (abs_x, abs_y), (abs_x+abs_w, abs_y+abs_h), (0,255,0), 2)
                    cv2.putText(frame_display, name, (abs_x, abs_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                else:
                    # UNKNOWN: deduplicate using histogram comparison with templates
                    # compute normalized grayscale hist
                    hist = cv2.calcHist([face_gray], [0], None, HIST_SIZE, HIST_RANGE)
                    cv2.normalize(hist, hist)
                    matched = False
                    matched_idx = None
                    for idx, templ in enumerate(unknown_templates):
                        if templ['hist'] is None:
                            # If the template was spatial-only (no face hist), skip matching hist
                            # but if spatial near, treat as matched
                            if abs(templ.get('center_x', cx)-cx) < UNKNOWN_COARSE and abs(templ.get('center_y', cy)-cy) < UNKNOWN_COARSE and (now_t - templ['last_seen'] < UNKNOWN_EXPIRE_SEC):
                                matched = True
                                matched_idx = idx
                                break
                            continue
                        # histogram correlation (1.0 = identical)
                        score = cv2.compareHist(templ['hist'], hist, cv2.HISTCMP_CORREL)
                        if score >= UNKNOWN_MATCH_THRESH:
                            matched = True
                            matched_idx = idx
                            break

                    if matched:
                        # update last_seen of the matched template; do not increment count
                        templ = unknown_templates[matched_idx]
                        templ['last_seen'] = now_t
                    else:
                        # new unknown person -> add template & increment
                        unknown_templates.append({'hist': hist, 'first_seen': now_t, 'last_seen': now_t, 'center_x': cx, 'center_y': cy})
                        unknown_total += 1
                        print(f"New unknown person counted: total unknown = {unknown_total}")

                    # draw unknown face box
                    cv2.rectangle(frame_display, (abs_x, abs_y), (abs_x+abs_w, abs_y+abs_h), (0,0,255), 2)
                    cv2.putText(frame_display, "Unknown", (abs_x, abs_y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # === expire old unknown templates periodically ===
        now_t = time.time()
        unknown_templates = [t for t in unknown_templates if now_t - t['last_seen'] <= UNKNOWN_EXPIRE_SEC]

        # overlay counts
        status = f"Known: {known_total}  Unknown: {unknown_total}  Vehicles: {vehicle_total}"
        cv2.putText(frame_display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Recognition - press q to quit", frame_display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Session summary:")
    print(f"  Known people: {known_total}")
    print(f"  Unknown people: {unknown_total}")
    print(f"  Vehicles: {vehicle_total}")
    print(f"Attendance saved to '{ATTENDANCE_CSV}'")

# 4) Show attendance
def show_attendance():
    if not os.path.exists(ATTENDANCE_CSV):
        print("No attendance records found.")
        return
    df = pd.read_csv(ATTENDANCE_CSV)
    if df.empty:
        print("No attendance records found.")
        return
    print(df.to_string(index=False))

# Simple CLI menu
def menu():
    print("\nCampus Security System")
    print("1. Capture images")
    print("2. Train model")
    print("3. Recognize & Mark")
    print("4. Show details")
    print("5. Exit")

    choice = input("Choose an option [1-5]: ")
    return choice.strip()

if __name__ == '__main__':
    while True:
        choice = menu()
        if choice == '1':
            capture_images()
        elif choice == '2':
            train_model()
        elif choice == '3':
            mark_attendance()
        elif choice == '4':
            show_attendance()
        elif choice == '5':
            print("Goodbye")
            break
        else:
            print("Invalid choice. Try again.")
