import cv2
import numpy as np
from picamera2 import Picamera2
from flask  import  Flask, Response

app = Flask(__name__)


# Setup Night Vision camera
tuning = Picamera2.load_tuning_file("ov5647_noir.json")
camera = Picamera2(tuning=tuning)
camera.configure(camera.create_preview_configuration())
camera.set_controls({"Saturation": -1.0})


frame = None

def generate_frames():
    global frame
    while True:
        # Capture a frame using Picamera2
        frame = camera.capture_array()

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    camera.start()
    app.run(host='0.0.0.0', port=5000)
