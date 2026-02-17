# AI-Powered Smart Surveillance for Campus Security and Monitoring

An intelligent AI-based surveillance system designed for automated campus security, real-time human and vehicle detection, tracking, and person identification with logging.  
The system integrates deep learning object detection, multi-object tracking, and facial recognition to enhance campus monitoring and safety.

---

## 📌 Project Overview

This project provides a smart surveillance solution that can:

- Detect people and vehicles in real-time video streams  
- Track individuals and vehicles across frames  
- Recognize known persons using face recognition  
- Log detected person details with timestamp  
- Count and log human and vehicle movement activity  
- Generate monitoring outputs for security analysis  

It is suitable for:

- Campus security monitoring  
- Parking and vehicle monitoring  
- Restricted area surveillance  
- Crowd and traffic monitoring  
- Smart premises monitoring  

---

## 🚀 Features

- Real-time human and vehicle detection using YOLOv8 / MobileNetSSD  
- Multi-object tracking with ByteTrack  
- Face recognition-based person identification  
- Automatic person detection logging (ID, name, date, time)  
- People and vehicle counting and movement logging  
- Video processing and detection output generation  
- Integrated AI surveillance pipeline  

---

## 🧠 Technologies Used

- Python  
- OpenCV  
- YOLOv8  
- MobileNetSSD  
- ByteTrack  
- Haar Cascade (Face Detection)  
- Deep Learning Models  
- Computer Vision  

---

## 📂 Project Structure

```
AI-Powered-Smart-Surveillance/
│
├── app.py                     # Main surveillance & detection application
├── face_attendance.py         # Face recognition & person logging module
├── ByteTrack/                 # Multi-object tracking module
├── dataset/                   # Face dataset
├── yolov8s.pt                 # YOLOv8 detection model
├── MobileNetSSD_deploy.*      # MobileNetSSD model files
├── haarcascade_frontalface_default.xml
├── trainer.yml                # Face recognition training data
├── Attendance.csv             # Logged person detection records
├── counts_log.csv             # Movement logs
├── detected_output_video.mp4  # Sample output
└── major report.pdf           # Project report
```

---

## ⚙️ System Workflow

1. Video input captured from camera or file  
2. Human and vehicle detection using YOLOv8 / MobileNetSSD  
3. Multi-object tracking using ByteTrack  
4. Face detection and recognition for known persons  
5. Logging of detected person details (ID, name, date, time)  
6. Human and vehicle movement counting and monitoring  
7. Output video generation and logs  

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install opencv-python ultralytics numpy pandas
```

### 2️⃣ Run surveillance system

```bash
python app.py
```

### 3️⃣ Run face recognition logging module

```bash
python face_attendance.py
```

---

## 📊 Output

- Detected & tracked human and vehicle video output  
- Logged person detection CSV file  
- People and vehicle count logs  
- Face recognition results  

---

## 🎯 Applications

- Smart campus surveillance  
- Parking and vehicle monitoring  
- Office and public security monitoring  
- Restricted zone detection  
- Smart premises monitoring  

---

## 🔮 Future Enhancements

- Real-time alert notifications  
- Intrusion and anomaly detection  
- Weapon detection  
- Cloud monitoring dashboard  
- Web-based control panel  

---

## 👩‍💻 Author

**Kumuda DP**  
AI & Computer Vision Enthusiast  

GitHub: https://github.com/Kumudadp
