🚗 AI-Powered Driver Safety System
Real-Time Drowsiness Detection + Phone Detection Using OpenCV, Dlib, and Deep Learning

This project is an Advanced Driver Monitoring System that detects:

Drowsiness / Sleepiness

Yawning

Eyes Closed

Head Down Movement

Mobile Phone Usage While Driving

and provides real-time voice alerts such as:
✔️ “Do not sleep while driving.”
✔️ “Do not use phone while driving.”

Built using Computer Vision, Facial Landmark Detection, and Deep Learning Object Detection.

This project is designed for road safety, automotive AI, and ADAS (Advanced Driver Assistance Systems).

🧭 Table of Contents

Overview

Features

Tech Stack Used

How It Works

Project Architecture

Installation

Run the Project

Future Improvements

Why Recruiters Will Love This Project

Contact

🔍 Overview

Road accidents due to driver drowsiness and mobile phone distraction are increasing every day.
This project aims to solve this real-world problem using:

Eye Aspect Ratio (EAR)

Mouth Aspect Ratio (MAR)

Head Pose Estimation

Deep Learning Phone Detection (COCO SSD Model)

Whenever the system detects unsafe behaviour, it gives voice alerts and displays warnings on screen.

✨ Features
🧠 1. Drowsiness Detection

Detects if the driver is:

Closing eyes for too long

Yawning

Looking down (head tilt detection)

📵 2. Mobile Phone Detection

Uses SSD MobileNet model to detect phone usage in real time.

🔊 3. Smart Voice Alerts

Different alert for different conditions:

Sleep Alert: “Do not sleep while driving.”

Phone Alert: “Do not use phone while driving.”

⚡ 4. Real-Time Processing

Optimized for 30+ FPS on most laptops.

🛠️ 5. Easy to Configure

Adjustable thresholds

Platform-independent

Minimal dependencies

🧑‍💻 Tech Stack Used
Category	Technologies
Languages	Python
Computer Vision	OpenCV
Facial Landmark Detection	Dlib (68 Landmark Model)
Deep Learning Model	SSD MobileNet (COCO, Class ID 77 for Phone)
Math & Processing	NumPy
Voice Output	macOS say() API / Windows Beep
🧠 How It Works
1️⃣ Facial Landmark Extraction

Detects 68 landmark points using Dlib

Computes EAR, MAR, and head tilt

2️⃣ Drowsiness Logic

If:

Eyes closed for N frames

Yawn detected

Head tilt > threshold

Then → trigger sleep alert

3️⃣ Phone Detection

Runs deep learning model on frame

If phone is detected consecutively for M frames
Then → trigger phone alert

4️⃣ Voice Alerts

Uses platform-specific TTS or beeps.

PythonProject/
│
├── .venv/                             # Virtual environment (auto-created)
│   ├── bin/
│   ├── lib/
│   ├── pyvenv.cfg
│   └── ...
│
├── drowsiness_dlib.py                 # Main driver monitoring script
├── shape_predictor_68_face_landmarks.dat   # Dlib facial landmark model
│
└── .gitignore                         # Git ignore rules

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/your-username/driver-safety-system.git
cd driver-safety-system

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Download required models

Dlib Landmark Model

MobileNet SSD Model + Config

Place them in the correct folders.

▶️ Run the Project
python driver_monitor.py


Press Q to quit.


	
🚀 Future Improvements

Alarm vibration for car seat

Night mode IR support

Deep learning–based eye state classifier

Integration with IoT (send alert to cloud)


Computer Vision

Deep Learning

Data Processing

Real-time Systems

Human Safety Engineering

✔ Demonstrates practical engineering skills

Including optimization, architecture design, and multi-sensor logic.

✔ Perfect for roles in:

AI / ML Engineer

Computer Vision Engineer

Automotive AI

Robotics

Embedded Systems

Research Engineer

📞 Contact

Developer: Om Raj
📧 Email: omraj6666@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/om-raj-vit/

🐙 GitHub:https://github.com/OmRaj6666
