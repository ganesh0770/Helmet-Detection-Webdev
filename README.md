🪖 AI Helmet Detection Web App

This project implements a real-time helmet detection system using YOLOv3 and OpenCV.
The system is deployed as a web application for browser-based monitoring, enabling real-time detection of helmet usage for safety compliance.


Tech Stack

Deep Learning / Computer Vision: YOLOv3, OpenCV, Python

Backend: FastAPI for API serving

Frontend: Next.js for the web interface

Deployment: Browser-accessible web application



# Project Structure
Helmet-DetectionWebdev/
│
├── backend/
│   ├── main.py          # FastAPI backend
│   ├── yolov3model/        # YOLOv3 detection module
│   ├── datadb.py
│   ├── models/
│   ├── viewdb.py
│   └── requirements.txt
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── pages/           # Next.js pages
│   ├── public/          # Static assets
│   └── package.json
│
│







Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/ganesh0770/Helmet-DetectionWebdev.git

1.cd frontend
2.npm install
3.npm tun dev
Open http://localhost:3000 in your browser to access the web app.




2️⃣ Install Backend Dependencies

# windows 
1.cd backend
2.python -m venv venv
3.venv\Scripts\Activate.ps1 
4.pip install -r requirements.txt
5.pip install "fastapi[standard]"
6.fastapi dev main.py

# In linux other than debian based(if you're using nix os)

1.cd backend
2.python -m venv venv
3.source venv/bin/activate
4.(if any conflict between package and dependancy)
use nix-shell(shell.nix)
5.pip install -r requirements.txt
6.pip install "fastapi[standard]"
7.fastapi dev main.py



 

#
