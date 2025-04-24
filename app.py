from flask import Flask, Response, request, render_template, redirect, url_for, session, jsonify
import cv2
import numpy as np
from datetime import datetime
import threading
import time
import os
from datetime import timedelta
import warnings
from camera import VideoCamera
from ultralytics import YOLO
import subprocess


app = Flask(__name__)
app.secret_key = 'rmcscamerasystem'

# AI Parameters
CONFIDENCE_THRESHOLD = 0.6
TARGET_CLASSES = ['person', 'car','bicycle','motorcycle','bus','train','truck']
GREEN = (0, 255, 0)
RED = (0, 0, 255)

model = YOLO("yolov8n.pt")
# Enable cuda if present
# model.to('cuda')
class_names = model.names
# Admin credentials
ADMIN_USERNAME = "admin"
PASS = "admin123"
camera = VideoCamera()

# SETUP PATH
current_path = os.path.abspath(__file__)
STORAGE = os.path.splitdrive(current_path)[0]

# Default Video Settings
video_settings = {
    "brightness": 0,
    "contrast": 0,
}

# Global Variables
recording_continuous = False
recording_motion = False
recording_lookout = False
lookout_mode = False
lookout_highlight_enabled = False
wr_frame = None
last_frame = None
lookout_mode_enabled = False 
motion_highlight_enabled = False
motion_mode_enabled = False
describe_frame = None
voice = 0
notify = 0

app_start_time = time.time()

def stop_all_modes():
    global recording_continuous, recording_motion, recording_lookout, motion_mode_enabed
    recording_continuous = False
    recording_motion = False
    recording_lookout = False
    lookout_mode = False
    motion_mode_enabed = False
    print("[+] All modes stopped.")

# Function Continuous recording
def record_video():
    CLIPLENGTH = 1800  
    global recording_continuous, wr_frame

    folder_name = datetime.now().strftime("%d-%m-%Y")
    folder_path = os.path.join(f"{STORAGE}\\CCTV\\", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    start_time = time.time()

    while recording_continuous:
        if wr_frame is not None:
            if out is None:
                filename = datetime.now().strftime("%H-%M-%S") + ".mp4"
                filepath = os.path.join(folder_path, filename)
                height, width, _ = wr_frame.shape
                out = cv2.VideoWriter(filepath, fourcc, 15, (width, height))

            out.write(wr_frame)

            if time.time() - start_time >= CLIPLENGTH:
                out.release()
                out = None
                start_time = time.time()

        time.sleep(1 / 20.0)

    if out:
        out.release()

# Motion Recording 

def motion_mode_recording():
    global wr_frame, recording_motion, motion_mode_enabled, last_frame, voice, notify

    clip_length = 30 
    folder_base = f"{STORAGE}\\CCTVMOTION"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while motion_mode_enabled:
        frame = wr_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        # --- Motion Detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if last_frame is None:
            last_frame = gray
            continue

        frame_delta = cv2.absdiff(last_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = any(cv2.contourArea(c) > 500 for c in contours)
        last_frame = gray

        if motion_detected and not recording_motion:
            print("[+] Motion detected, starting recording.")
            if voice == 1 or notify == 1:
                describe_frame = frame.copy()
                img_folder_base = f"{STORAGE}\\CCTVIMAGE"
                img_folder_name = datetime.now().strftime("%d-%m-%Y")
                img_folder_path = os.path.join(img_folder_base, img_folder_name)
                os.makedirs(img_folder_path, exist_ok=True)

                img_filename = datetime.now().strftime("%H-%M-%S") + ".jpg"
                img_file_path = os.path.join(img_folder_path, img_filename)
                cv2.imwrite(img_file_path, describe_frame)
                print(f"[+] Full Image Path: {img_file_path}")

                DESCRIBE_APP = f"{STORAGE}\\APP\\describe.py"
                #Implement ai describe
                print("[+] Describing Image")
                process = subprocess.Popen(['python3',DESCRIBE_APP,img_file_path,str(voice),str(notify)])
            
            recording_motion = True

            folder_name = datetime.now().strftime("%d-%m-%Y")
            folder_path = os.path.join(folder_base, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            filename = datetime.now().strftime("%H-%M-%S") + "_motion.mp4"
            filepath = os.path.join(folder_path, filename)

            height, width, _ = frame.shape
            out = cv2.VideoWriter(filepath, fourcc, 15, (width, height))  

            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < clip_length and motion_mode_enabled:
                frame = wr_frame.copy()
                if frame is not None:
                    out.write(frame)
                    frame_count += 1

                time.sleep(1 / 15) 

            out.release()
            recording_motion = False
            print("[+] Motion recording finished.")
            print(f"[+] Total frames written: {frame_count}")




def lookout_mode_recording():
    global wr_frame, recording_lookout, lookout_mode_enabled, lookout_frame, voice, notify

    clip_length = 30  
    folder_base = f"{STORAGE}\\CCTVLOOKOUT"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    confidence = 0

    frame_skip_rate = 15  
    frame_counter = 0

    while lookout_mode_enabled:
        frame = wr_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        detected = False

        if frame_counter % frame_skip_rate == 0:
            results = model(frame, verbose=False)[0]

            for data in results.boxes.data.tolist():
                xmin, ymin, xmax, ymax = map(int, data[:4])
                confidence = float(data[4])
                class_id = int(data[5])
                label = class_names[class_id]

                if label in TARGET_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                    detected = True
                    break 

        lookout_frame = frame
        frame_counter += 1

        if detected and not recording_lookout:


            # if confidence >= 0.70:
            if 1 == 1:
                print(f"[+] Confidence: {confidence}")
                describe_frame = frame.copy()
                img_folder_base = f"{STORAGE}\\CCTVLOOKOUTIMAGE"
                img_folder_name = datetime.now().strftime("%d-%m-%Y")
                img_folder_path = os.path.join(img_folder_base, img_folder_name)
                os.makedirs(img_folder_path, exist_ok=True)

                img_filename = datetime.now().strftime("%H-%M-%S") + ".jpg"
                img_file_path = os.path.join(img_folder_path, img_filename)
                cv2.imwrite(img_file_path, describe_frame)
                print(f"[+] Full Image Path: {img_file_path}")

                DESCRIBE_APP = f"{STORAGE}\\APP\\describe.py"
                #Implement ai describe
                print("[+] Describing Image")
                process = subprocess.Popen(['python3',DESCRIBE_APP,img_file_path,str(voice),str(notify)])
            

            print("[+] Target detected, starting lookout recording.")
            recording_lookout = True

            folder_name = datetime.now().strftime("%d-%m-%Y")
            folder_path = os.path.join(folder_base, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            filename = datetime.now().strftime("%H-%M-%S") + ".mp4"
            filepath = os.path.join(folder_path, filename)

            height, width, _ = frame.shape
            out = cv2.VideoWriter(filepath, fourcc, 15, (width, height))  

            start_time = time.time()
            frame_count = 0

            while time.time() - start_time < clip_length and lookout_mode_enabled:
                frame = wr_frame.copy()
                if frame is not None:
                    out.write(frame)
                    frame_count += 1

                time.sleep(1 / 60)  

            out.release()
            recording_lookout = False
            print("[+] Lookout recording finished.")
            print(f"[+] Total frames written: {frame_count}")

# Function to generate frames from the camera
def generate_frames():
    global last_frame, wr_frame, lookout_highlight_enabled, recording_lookout, lookout_frame, lookout_mode_enabled, motion_hilight_enabled, video_settings
    global lookout_lock, lookout_frame, camera
    FRAME_SKIP = 5  
    frame_count = 0
    mode_text = None
    while True:
        success, frame = camera.get_frame()

        if not success:
            print("[-] Camera Read Error: Restarting stream...")
            try:
                camera.__del__()  
            except Exception as e:
                print(f"Error releasing camera: {e}")

            time.sleep(3) 

            try:
                camera = VideoCamera() 
                print("[+] Camera restarted successfully.")
                continue
            except Exception as e:
                print(f"[!] Failed to restart camera: {e}")
                time.sleep(5)
                continue


        # ADD mode text
        if recording_continuous:
            mode_text = "REC"
        elif motion_mode_enabled:
            mode_text = "MOTION"
        elif lookout_mode_enabled:
            mode_text = "AI"
        else:
            mode_text = "None"

        frame_count += 1
        frame = cv2.convertScaleAbs(frame, alpha=video_settings["contrast"], beta=video_settings["brightness"])
        current_time = datetime.now().strftime(" %d-%m-%Y %H:%M:%S %a %p")
        cv2.putText(frame, current_time, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)


        # Display the mode in the top right corner
        if mode_text:
            text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10
            text_y = 25
            cv2.rectangle(frame, (text_x - 5, text_y - 20), (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, mode_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ai_frame = frame.copy()
          # --- MOTION DETECTION HILIGHT---
        if motion_highlight_enabled:
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
              
            last_frame = gray
        # -----AI Human and car Hilight ----
        if lookout_highlight_enabled:
            detected_objects = []
            results = model(frame, verbose=False)[0]
            # Analyze all detections
            for data in results.boxes.data.tolist():
                xmin, ymin, xmax, ymax = map(int, data[:4])
                confidence = float(data[4])
                class_id = int(data[5])
                label = class_names[class_id]

                # Check confidence 
                if label in TARGET_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                    detected_objects.append((label, (xmin, ymin, xmax, ymax), confidence))
                    color = GREEN if label == 'person' else RED
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        

        # ---------------TEST------------------

        # if lookout_highlight_enabled:
        #     frame = ai_detect(frame)
        
        wr_frame = frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame)

              
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

@app.route('/start_recording', methods=['POST'])
def start_recording():
    stop_all_modes() 

    global recording_continuous
    if not recording_continuous:
        recording_continuous = True
        threading.Thread(target=record_video).start()
    return ('', 204)

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording_continuous
    recording_continuous = False
    return ('', 204)


@app.route('/start_motion_recording', methods=['POST'])
def start_motion_recording():
    stop_all_modes()  

    global motion_mode_enabled, recording_motion
    motion_mode_enabled = True
    recording_motion = False  

    threading.Thread(target=motion_mode_recording).start()
    return ('', 204)


@app.route('/stop_motion_recording', methods=['POST'])
def stop_motion_recording():
    global motion_mode_enabled, recording_motion
    motion_mode_enabled = False
    recording_motion = False  
    return ('', 204)

@app.route('/start_lookout', methods=['POST'])
def start_lookout():
    stop_all_modes()  

    global lookout_mode, lookout_mode_enabled

    
    if not lookout_mode_enabled:
        lookout_mode_enabled = True
        threading.Thread(target=lookout_mode_recording).start()

    return ('', 204)


@app.route('/stop_lookout', methods=['POST'])
def stop_lookout():
    global lookout_mode_enabled, recording_lookout, lookout_highlight_enabled
    lookout_mode_enabled = False
    recording_lookout = False
    lookout_highlight_enabled = True  

    return ('', 204)

@app.route('/toggle_motion_detection', methods=['POST'])
def toggle_motion_highlight():
    global lookout_highlight_enabled, motion_highlight_enabled
    lookout_highlight_enabled = False
    motion_highlight_enabled = not motion_highlight_enabled
    print(f"[+] motion highlight: {motion_highlight_enabled}")
    return '', 204

@app.route('/toggle_lookout_highlight', methods=['POST'])
def toggle_lookout_highlight():
    global lookout_highlight_enabled, motion_highlight_enabled
    motion_highlight_enabled = False
    lookout_highlight_enabled = not lookout_highlight_enabled
    print(f"[+] Lookout highlight: {lookout_highlight_enabled}")
    return '', 204

@app.route('/uptime')
def uptime():
    uptime_seconds = time.time() - app_start_time
    return f"Uptime: {str(timedelta(seconds=int(uptime_seconds)))}"
# Implement Download
@app.route('/download')
def download():
    return render_template('download.html')

@app.route('/update_video_settings', methods=['POST'])
def update_video_settings():
    global video_settings
    data = request.get_json()
    
    video_settings["brightness"] = int(data.get("brightness", 0))
    video_settings["contrast"] = float(data.get("contrast", 0))

    return jsonify(success=True)

@app.route('/toggle_features', methods=['POST'])
def toggle_features():
    global voice, notify
    data = request.get_json()
    voice = int(data.get('voice', 0))
    notify = int(data.get('notify', 0))
    return jsonify({'voice': voice, 'notify': notify})

if __name__ == '__main__':
    
    app.run(host='127.0.0.1', port=5000, threaded=True)

