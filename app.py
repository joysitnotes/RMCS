from flask import Flask, Response, request, render_template, redirect, url_for, session, jsonify, send_from_directory, abort
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
import torch  
from flask_cors import CORS
from aiortc import RTCPeerConnection, RTCSessionDescription
import json
import uuid
import asyncio
import logging
import signal
import sys

app = Flask(__name__)
app.secret_key = 'rmcscamerasystem'

# Set to keep track of RTCPeerConnection instances
pcs = set()

#Privacy Mask
CORS(app)
mask_coordinates = []
privacy = False
mask = False
MOTION_THRESHOLD = 500

# AI Parameters
CONFIDENCE_THRESHOLD = 0.6
TARGET_CLASSES = ['person', 'car','bicycle','motorcycle','bus','train','truck']
SELECTED_CLASSES = TARGET_CLASSES
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# Model settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = YOLO("yolov8n.pt")
if device == 'cuda':
    model.to('cuda')

class_names = model.names
# admin credentials
ADMIN_USERNAME = "admin"
PASS = "admin123"

# For testing with webcam camType == 2 for Remote WS camera camType == 1
camType = 1
if camType == 1: 
    camera = VideoCamera()
else:
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 15)

# SETUP PATH
current_path = os.path.abspath(__file__)
STORAGE = os.path.splitdrive(current_path)[0]

DIGESTED_DIR = f"{STORAGE}\\CCTVDIGEST"
CCTVCLIPS = f"{STORAGE}\\CCTVCLIPS" 

# Default Video Settings
video_settings = {
    "brightness": 0,
    "contrast": 1.5,
}

# Global flags for different modes
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
digestday = ""

app_start_time = time.time()

def stop_all_modes():
    global recording_continuous, recording_motion, recording_lookout, motion_mode_enabed
    recording_continuous = False
    recording_motion = False
    recording_lookout = False
    lookout_mode = False
    motion_mode_enabed = False
    print("[+] All modes stopped.")


def signal_handler(sig, frame):
    stop_all_modes()  
    print("Exiting.....")
    sys.exit(0)  # Exit the program

def reset_camera():
    stop_all_modes()
    print("[*] All Modes Stopped")
    global camera
    try:
        camera.restart()() 
    except Exception as e:
        print(f"Error releasing camera: {e}")

    time.sleep(3) 

    try:
        camera = VideoCamera() 
        print("[+] Camera restarted successfully.")
    except Exception as e:
        print(f"[!] Failed to restart camera: {e}")
        time.sleep(5)


def cctv_digest(folder_name):
    global digestday
    DIGETSCCTVAPP = f"{STORAGE}\\APP\\CCTVDIGEST\\cctv_digest2.py"
    print(f"[+] Starting CCTV Digest on day: {folder_name}")
    process = subprocess.Popen(['python3',DIGETSCCTVAPP,str(folder_name)])


# Function to handle continuous recording
def record_video():
    CLIPLENGTH = 1800  # 30 minutes
    global recording_continuous, wr_frame

    folder_name = datetime.now().strftime("%d-%m-%Y")
    folder_path = os.path.join(f"{STORAGE}\\CCTV\\", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'H264')
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
    global wr_frame, recording_motion, motion_mode_enabled, last_frame, voice, notify, mask, MOTION_THRESHOLD

    clip_length = 30  # seconds
    folder_base = f"{STORAGE}\\CCTVCLIPS"
    fourcc = cv2.VideoWriter_fourcc(*'H264')

    while motion_mode_enabled:
        frame = wr_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue
        
        if mask:
            for coord in mask_coordinates:
                x, y = coord['x'], coord['y']
                cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 0, 0), -1)
        

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

        motion_detected = any(cv2.contourArea(c) > MOTION_THRESHOLD for c in contours)
        last_frame = gray

        if motion_detected and not recording_motion:
            print("[+] Motion detected, starting recording.")
            if voice == 1 or notify == 1:
                describe_frame = wr_frame.copy()
                img_folder_base = f"{STORAGE}\\CCTVIMAGE"
                img_folder_name = datetime.now().strftime("%d-%m-%Y")
                img_folder_path = os.path.join(img_folder_base, img_folder_name)
                os.makedirs(img_folder_path, exist_ok=True)

                img_filename = datetime.now().strftime("%H-%M-%S") + ".jpg"
                img_file_path = os.path.join(img_folder_path, img_filename)
                cv2.imwrite(img_file_path, describe_frame)
                print(f"[+] Full Image Path: {img_file_path}")

                DESCRIBE_APP = f"{STORAGE}\\APP\\AIDESCRIBE\\describe.py"
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
    global wr_frame, recording_lookout, lookout_mode_enabled, lookout_frame, voice, notify, mask, CONFIDENCE_THRESHOLD, SELECTED_CLASSES

    clip_length = 30  # seconds
    folder_base = f"{STORAGE}\\CCTVCLIPS"
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    confidence = 0

    frame_skip_rate = 15  
    frame_counter = 0

    while lookout_mode_enabled:
        frame = wr_frame.copy()
        if frame is None:
            time.sleep(0.05)
            continue

        if mask:
            for coord in mask_coordinates:
                x, y = coord['x'], coord['y']
                cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 0, 0), -1)
        detected = False

        # Skip some frames for detection to save CPU
        if frame_counter % frame_skip_rate == 0:
            results = model(frame, verbose=False)[0]

            for data in results.boxes.data.tolist():
                xmin, ymin, xmax, ymax = map(int, data[:4])
                confidence = float(data[4])
                class_id = int(data[5])
                label = class_names[class_id]

                if label in SELECTED_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                    detected = True
                    break 

        lookout_frame = frame
        frame_counter += 1

        if detected and not recording_lookout:


            if voice == 1 or notify == 1:
                print(f"[+] Confidence: {confidence}")
                describe_frame = wr_frame.copy()
                img_folder_base = f"{STORAGE}\\CCTVIMAGE"
                img_folder_name = datetime.now().strftime("%d-%m-%Y")
                img_folder_path = os.path.join(img_folder_base, img_folder_name)
                os.makedirs(img_folder_path, exist_ok=True)

                img_filename = datetime.now().strftime("%H-%M-%S") + ".jpg"
                img_file_path = os.path.join(img_folder_path, img_filename)
                cv2.imwrite(img_file_path, describe_frame)
                print(f"[+] Full Image Path: {img_file_path}")

                DESCRIBE_APP = f"{STORAGE}\\APP\\AIDESCRIBE\\describe.py"
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
            out = cv2.VideoWriter(filepath, fourcc, 30, (width, height))  

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






def generate_frames():
    global last_frame, wr_frame, lookout_highlight_enabled, recording_lookout
    global lookout_mode_enabled, motion_highlight_enabled, video_settings
    global lookout_lock, lookout_frame, camera, privacy, mask_coordinates
    global MOTION_THRESHOLD, CONFIDENCE_THRESHOLD, class_names, model, motion_mode_enabed

    FRAME_SKIP = 2
    frame_count = 0
    last_gray = None
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

    while True:
        if camType == 1:        
            success, frame = camera.get_frame()
        else:
            success, frame = camera.read()

        if not success:
            print("[-] Camera Read Error: Restarting stream...")
            try:
                camera.restart()
                time.sleep(2)
              
                continue
            except Exception as e:
                print(f"[!] Camera restart failed: {e}")
                time.sleep(5)
                continue
        

        contrast = video_settings.get("contrast", 1.5)
        brightness = video_settings.get("brightness", 0)
        frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

        # Privacy masking
        if privacy and mask_coordinates:
            for coord in mask_coordinates:
                x, y = coord['x'], coord['y']
                cv2.rectangle(frame, (x - 50, y - 50), (x + 50, y + 50), (0, 0, 0), -1)

        # Timestamp and mode
        mode_text = "Mode"
        if recording_continuous:
            mode_text = "REC"
        elif motion_mode_enabled:
            mode_text = "MOTION"
        elif lookout_mode_enabled:
            mode_text = "AI"

        current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S %a %p")
        cv2.putText(frame, current_time, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if mode_text:
            text_size = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10
            text_y = 25
            cv2.rectangle(frame, (text_x - 5, text_y - 20), 
                                 (text_x + text_size[0] + 5, text_y + 5), 
                                 (0, 0, 0), -1)
            cv2.putText(frame, mode_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Motion detection
        if motion_highlight_enabled and frame_count % FRAME_SKIP == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            if last_gray is None:
                last_gray = gray
            else:
                delta = cv2.absdiff(last_gray, gray)
                thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
                thresh = cv2.dilate(thresh, None, iterations=2)
                contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for c in contours:
                    if cv2.contourArea(c) < MOTION_THRESHOLD:
                        continue
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                last_gray = gray

        # AI detection
        if lookout_highlight_enabled and frame_count % FRAME_SKIP == 0:
            results = model(frame, verbose=False)[0]
            for data in results.boxes.data.tolist():
                xmin, ymin, xmax, ymax = map(int, data[:4])
                confidence = float(data[4])
                class_id = int(data[5])
                label = class_names[class_id]

                if label in SELECTED_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                    color = (0, 255, 0) if label == 'person' else (0, 0, 255)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

     
        wr_frame = frame.copy()

        # Encode for MJPEG
        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
        if not ret:
            continue
        frame = buffer.tobytes()
        frame_count += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



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
    return render_template('index.html', TARGET_CLASSES=TARGET_CLASSES, SELECTED_CLASSES=SELECTED_CLASSES)


@app.route('/start_recording', methods=['POST'])
def start_recording():
    stop_all_modes()  # Stop all modes before starting a new one

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
    stop_all_modes()  # Stop any existing modes

    global motion_mode_enabled, recording_motion
    motion_mode_enabled = True
    recording_motion = False  

    threading.Thread(target=motion_mode_recording).start()
    return ('', 204)


@app.route('/stop_motion_recording', methods=['POST'])
def stop_motion_recording():
    global motion_mode_enabled, recording_motion
    motion_mode_enabled = False
    recording_motion = False  # Stop recording if running
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
    lookout_highlight_enabled = False  

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
    uptime_str = str(timedelta(seconds=int(uptime_seconds)))
    return jsonify(uptime=uptime_str)

@app.route('/download')
def download():
    return render_template('download.html')



@app.route('/update_video_settings', methods=['POST'])
def update_video_settings():
    global video_settings, CONFIDENCE_THRESHOLD, MOTION_THRESHOLD
    data = request.get_json()
    
    video_settings["brightness"] = int(data.get("brightness", 0))
    video_settings["contrast"] = float(data.get("contrast", 0))
    
    CONFIDENCE_THRESHOLD = float(data.get("aiConfidence", 60)) / 100.0 
    MOTION_THRESHOLD = int(data.get("motionSensitivity", 500)) 

    print(f"Updated: Brightness={video_settings['brightness']}, Contrast={video_settings['contrast']}, "
          f"AIConfidence={CONFIDENCE_THRESHOLD}, MotionThreshold={MOTION_THRESHOLD}")
    
    return jsonify(success=True)


@app.route('/toggle_features', methods=['POST'])
def toggle_features():
    global voice, notify
    data = request.get_json()
    voice = int(data.get('voice', 0))
    notify = int(data.get('notify', 0))
    return jsonify({'voice': voice, 'notify': notify})


@app.route('/reset_camera', methods=['POST'])
def camera_reset():
    reset_camera()
    return '',204



@app.route('/digest')
def digest():
 
    directory_path = f"{STORAGE}\\CCTV" 
 
    folders = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]
    
    return render_template('digest.html', folders=folders)

@app.route('/day/<folder_name>')
def day(folder_name):
    cctv_digest(folder_name)
   
    return redirect(url_for('index'))


@app.route("/set_mask", methods=["POST"])
def set_mask():
    global privacy, mask, mask_coordinates
    data = request.json
    mask_coordinates = data['coordinates']
    privacy = data['privacy']
    mask = True
    print(f"[+] Received coordinates")
    print(f"[+] Privacy mask status: {privacy}")
    print(f"[+] Mask status: {mask}")
    
    return {"status": "success"}

@app.route("/get_mask", methods=["GET"])
def get_mask():
    return jsonify({"coordinates": mask_coordinates})

# Route to disable the privacy mask
@app.route("/disable_mask", methods=["POST"])
def disable_mask():
    global privacy, mask, mask_coordinates
    privacy = False
    mask = False
    mask_coordinates = []  
    print("[+] Privacy mask disabled.")
    
    return {"status": "success"}

# Asynchronous function to handle offer exchange
async def offer_async():
    params = await request.json
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    # Create an RTCPeerConnection instance
    pc = RTCPeerConnection()

    # Generate a unique ID for the RTCPeerConnection
    pc_id = "PeerConnection(%s)" % uuid.uuid4()
    pc_id = pc_id[:8]


    # Create and set the local description
    await pc.createOffer(offer)
    await pc.setLocalDescription(offer)

    response_data = {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

    return jsonify(response_data)

def offer():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    future = asyncio.run_coroutine_threadsafe(offer_async(), loop)
    return future.result()

# Route to handle the offer request
@app.route('/offer', methods=['POST'])
def offer_route():
    return offer()

# Route to stream video frames
@app.route('/video_feed')
def video_feed():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Digested Routes
@app.route('/digested')
def digested():
    # List all date folders in the base directory
    date_folders = [f for f in os.listdir(DIGESTED_DIR) if os.path.isdir(os.path.join(DIGESTED_DIR, f))]
    return render_template('digested.html', date_folders=date_folders)

@app.route('/<date_folder>')
def show_date_folder(date_folder):
    date_folder_path = os.path.join(DIGESTED_DIR, date_folder)
    if not os.path.exists(date_folder_path) or not os.path.isdir(date_folder_path):
        return abort(404)

    # List all category folders
    category_folders = [f for f in os.listdir(date_folder_path) if os.path.isdir(os.path.join(date_folder_path, f))]
    return render_template('date_folder.html', date_folder=date_folder, category_folders=category_folders)

@app.route('/<date_folder>/<category_folder>')
def show_category_folder(date_folder, category_folder):
    category_folder_path = os.path.join(DIGESTED_DIR, date_folder, category_folder)
    if not os.path.exists(category_folder_path) or not os.path.isdir(category_folder_path):
        return abort(404)

    # List all images in the selected category folder
    images = [f for f in os.listdir(category_folder_path) if f.lower().endswith(('jpg', 'jpeg', 'png', 'gif'))]
    return render_template('category_folder.html', date_folder=date_folder, category_folder=category_folder, images=images)

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(DIGESTED_DIR, filename)


# CLIPS DISPLAY
@app.route('/clips')
def clips():
    # List all day folders
    day_folders = [f for f in os.listdir(CCTVCLIPS) if os.path.isdir(os.path.join(CCTVCLIPS, f))]
    return render_template('clips.html', day_folders=day_folders)

@app.route('/videos/<folder>')
def videos(folder):
    # List all video files in the selected day folder
    day_folder_path = os.path.join(CCTVCLIPS, folder)
    videos = [f for f in os.listdir(day_folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    return render_template('videos.html', folder=folder, videos=videos)

@app.route('/videos/<path:folder>/<path:filename>')
def video(folder, filename):
    # Serve the video file from the specified folder
    return send_from_directory(os.path.join(CCTVCLIPS, folder), filename)



@app.route('/submit', methods=['POST'])
def submit():
    global SELECTED_CLASSES
    SELECTED_CLASSES = request.form.getlist('items') 
    print("Selected items:", SELECTED_CLASSES) 
    return redirect(url_for('index'))

if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)
 
    app.run(host='127.0.0.1', port=5000, threaded=True)
