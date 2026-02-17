# AI-Powered Smart Surveillance for Campus Security and Monitoring

An intelligent AI-based surveillance system designed for automated campus security, real-time human detection, tracking, and face-based attendance management.  
The system integrates deep learning object detection, multi-object tracking, and facial recognition to enhance campus monitoring and safety.

---

## 📌 Project Overview

This project provides a smart surveillance solution that can:

- Detect people in real-time video streams  
- Track individuals across frames  
- Recognize faces for attendance logging  
- Count and log movement activity  
- Generate monitoring outputs for security analysis  

It is suitable for:

- Campus security monitoring  
- Classroom attendance automation  
- Restricted area surveillance  
- Crowd monitoring  

---

## 🚀 Features

- Real-time human detection using YOLOv8 / MobileNetSSD  
- Multi-object tracking with ByteTrack  
- Face recognition-based attendance system  
- Automated attendance CSV logging  
- People counting and movement logging  
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
├── app.py                     # Main surveillance application
├── face_attendance.py         # Face recognition attendance system
├── ByteTrack/                 # Multi-object tracking module
├── dataset/                   # Face dataset
├── yolov8s.pt                 # YOLOv8 detection model
├── MobileNetSSD_deploy.*      # MobileNetSSD model files
├── haarcascade_frontalface_default.xml
├── trainer.yml                # Face recognition training data
├── Attendance.csv             # Attendance records
├── counts_log.csv             # Movement logs
├── detected_output_video.mp4  # Sample output
└── major report.pdf           # Project report
```

---

## ⚙️ System Workflow

1. Video input captured from camera or file  
2. Person detection using YOLOv8 / MobileNetSSD  
3. Object tracking using ByteTrack  
4. Face detection and recognition  
5. Attendance logging  
6. Movement counting and monitoring  
7. Output video generation  

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

### 3️⃣ Run attendance module

```bash
python face_attendance.py
```

---

## 📊 Output

- Detected & tracked video output  
- Attendance CSV file  
- People count logs  
- Face recognition results  

---

## 🎯 Applications

- Smart campus surveillance  
- Classroom attendance automation  
- Office security monitoring  
- Public area monitoring  
- Restricted zone detection  

---

## 🔮 Future Enhancements

- Real-time alert notifications  
- Intrusion detection  
- Weapon detection  
- Cloud monitoring dashboard  
- Web-based control panel  

---

## 👩‍💻 Author

**Kumuda DP**  
AI & Computer Vision Enthusiast  

GitHub: https://github.com/Kumudadp
