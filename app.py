from flask import Flask, Response, request, render_template, redirect, url_for, session
import cv2
import numpy as np
from datetime import datetime
import subprocess
import threading
import time
import os

app = Flask(__name__)
app.secret_key = 'rmcscamerasystem'

# admin credentials
ADMIN_USERNAME = "admin"
PASS = "admin123"
# camera
camera = cv2.VideoCapture("http://192.168.137.158:5000/")


camera.set(cv2.CAP_PROP_FPS, 20)

recording = False
motion_detection_enabled = False
last_frame = None
wr_frame = None
recording = False

# Add motion only recording function 

def record_video():
    global recording, wr_frame

 
    folder_name = datetime.now().strftime("%d-%m-%Y")
    folder_path = os.path.join("D:\\", folder_name)
    os.makedirs(folder_path, exist_ok=True)


    filename = datetime.now().strftime("%H-%M-%S") + ".mp4"
    filepath = os.path.join(folder_path, filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None

    while recording:
        if wr_frame is not None:
            if out is None:
                height, width, _ = wr_frame.shape
                out = cv2.VideoWriter(filepath, fourcc, 20.0, (width, height))

            out.write(wr_frame)
        time.sleep(1 / 20.0) 

    if out:
        out.release()

def generate_frames():
    global last_frame
    global wr_frame
    while True:
        success, frame = camera.read()
        if not success:
            break

        if motion_detection_enabled:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if last_frame is None:
                last_frame = gray
                continue

            frame_delta = cv2.absdiff(last_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 500:
                    continue
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Motion Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            last_frame = gray

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, current_time, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        #ret, buffer = cv2.imencode('.jpg', frame)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        wr_frame = frame
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    username = request.form['username']
    password = request.form['password'].encode('utf-8')
    
    if username == ADMIN_USERNAME and PASS == "admin123":
        session['logged_in'] = True
        return redirect(url_for('index'))
    return "Invalid credentials!", 401

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

@app.route('/index')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_motion_detection', methods=['POST'])
def toggle_motion_detection():
    global motion_detection_enabled
    motion_detection_enabled = not motion_detection_enabled
    return ('', 204)


@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    if not recording:
        recording = True
        threading.Thread(target=record_video).start()
    return ('', 204)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    return ('', 204)

@app.route('/download')
def download():
    return render_template('download.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
