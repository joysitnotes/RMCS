
#!/usr/bin/python3

import asyncio
import websockets
import base64
from picamera2 import Picamera2
import cv2
import time
tuning = Picamera2.load_tuning_file("ov5647_noir.json")
picam2 = Picamera2(tuning=tuning)
picam2.video_configuration.controls.FrameRate = 15.0
picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))


picam2.start()
time.sleep(15)
print("[+] Camera started")


async def video_stream(websocket):
    print("[+] Client connected")
    try:
        while True:
            frame = picam2.capture_array("main")

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            frame_data = base64.b64encode(buffer).decode('utf-8')

            await websocket.send(frame_data)

          
            await asyncio.sleep(1 / 20)
    except websockets.exceptions.ConnectionClosedOK:
        print("[-] Connection closed by client")
    except asyncio.TimeoutError:
        print("[-] Connecetion timed out")
    except Exception as e:
        print(f"[-] Error: {e}")


async def main():
    async with websockets.serve(video_stream, "0.0.0.0", 8765, ping_interval=40):
        print("[+] WebSocket server started")
        await asyncio.Future()

if __name__ == "__main__":

    try:
        asyncio.run(main())
    finally:
        picam2.stop()
        print("[-] Camera stopped")
        print("[-] Exitting..")
