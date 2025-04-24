import pyttsx3
import os
import warnings
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import sys
from datetime import datetime
from pushbullet import Pushbullet
import mimetypes

# Add pushbullet api token
pb = Pushbullet("API_TOKEN")
now = datetime.now()

if len(sys.argv) != 4:
    print("[-] Not enough Arguments")
    sys.exit(1)
else:
    IMAGEPATH = sys.argv[1]
    VOICE = int(sys.argv[2])
    NOTIFY = int(sys.argv[3])

warnings.filterwarnings("ignore", message="Using a slow image processor as `use_fast` is unset")

# Load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate a detailed caption for a local image
def generated_caption(image_path):
    try:
        image = Image.open(image_path)

        
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=100)  
        caption = processor.decode(out[0], skip_special_tokens=True)

        

        return caption
    except Exception as e:
        return f"An error occurred: {e}"

caption = generated_caption(IMAGEPATH)
time = now.strftime("%H:%M %p")
full_caption = f"Warning On Camera 1 at {time} there is {caption}"

print(f"{full_caption}")



# Speech 
if VOICE == 1:
    print("[+] Voice Active")
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)   
    engine.setProperty('volume', 1)     
    engine.say(full_caption)
    engine.runAndWait()

# Notifications
if NOTIFY == 1:


    print("[+] Sending Notification")
    mime_type, _ = mimetypes.guess_type(IMAGEPATH)

    with open(IMAGEPATH, "rb") as f:
        file_data = pb.upload_file(f, IMAGEPATH)

    pb.push_file(file_name=file_data["file_name"],
                 file_url=file_data["file_url"],
                 file_type=mime_type,
                 body=full_caption,
                 title="WARNING: ")
