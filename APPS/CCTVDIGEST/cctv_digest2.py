import cv2
import os
import glob
import sys
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor
import imagehash
from PIL import Image
import torch
import time 
# SETUP PATH
current_path = os.path.abspath(__file__)
STORAGE = os.path.splitdrive(current_path)[0]
FOLDER_DAY = sys.argv[1]


CONFIDENCE_LIMIT = 0.6
OBJECT_CLASSES = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'train', 'truck']
FRAME_SKIP_SECONDS = 30
OUTPUT_DIRECTORY = f"{STORAGE}\\CCTVDIGEST\\{FOLDER_DAY}"
MODEL_PATH = 'yolov8n.pt'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

detector = YOLO(MODEL_PATH)
if device == 'cuda':
    detector.to('cuda')



def analyze_frame(frame):
    detections = set()
    result = detector(frame, verbose=False)[0]

    for detection in result.boxes.data.tolist():
        x1, y1, x2, y2 = map(int, detection[:4])
        confidence = float(detection[4])
        class_id = int(detection[5])
        label = detector.names[class_id]

        if label in OBJECT_CLASSES and confidence >= CONFIDENCE_LIMIT:
            detections.add(label)

    return frame, detections


def calculate_image_hash(image_path):
    image = Image.open(image_path)
    return imagehash.average_hash(image)


def remove_similar_images_in_folder(class_folder):
    print("[+] DELETING SIMILAR IMAGES")
    images = sorted(glob.glob(os.path.join(class_folder, "*.jpg")))
    hashes = {}
    to_delete = set()

    for image_path in images:
        img_hash = calculate_image_hash(image_path)
        if img_hash in hashes:
            to_delete.add(image_path) 
        else:
            hashes[img_hash] = image_path

   
    for image_path in to_delete:
        os.remove(image_path)
        print(f"[+] [DELETE] Removed duplicate: {image_path}")


def run_digest_mode(video_file):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps * FRAME_SKIP_SECONDS)
    frame_index = 0
    save_count = 1

    video_name = os.path.splitext(os.path.basename(video_file))[0]

    while True:
        success, frame = cap.read()
        if not success:
            break

        processed_frame, detected_labels = analyze_frame(frame)

        if detected_labels:
            for label in detected_labels:
                class_output_dir = os.path.join(OUTPUT_DIRECTORY, label)
                os.makedirs(class_output_dir, exist_ok=True)

                filename = f"{video_name}_{save_count}.jpg"
                save_path = os.path.join(class_output_dir, filename)
                cv2.imwrite(save_path, processed_frame)

            save_count += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index + skip_frames)
            frame_index += skip_frames
        else:
            frame_index += 1

    cap.release()
    print(f"Finished: {video_name} — {save_count - 1} frames saved")

    with ProcessPoolExecutor() as executor:
        for label in OBJECT_CLASSES:
            class_folder = os.path.join(OUTPUT_DIRECTORY, label)
            if os.path.exists(class_folder):
                executor.submit(remove_similar_images_in_folder, class_folder)

if __name__ == "__main__":

    start_time = time.time()
    video_folder = f"{STORAGE}\\CCTV\\{FOLDER_DAY}"
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))

    if not video_files:
        print("[-] No .mp4 files found in the specified folder.")
    else:
        for video_path in video_files:
            print(f"[*] Processing: {video_path}")
            run_digest_mode(video_path)

        print(" All videos processed.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.6f} seconds")
