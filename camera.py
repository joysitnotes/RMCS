import subprocess
import numpy as np
import cv2

class VideoCamera:
    def __init__(self):
        self.width = 640  
        self.height = 480

        # Add RTSP Camera URL, Password, IP and Port
        rtsp_url = "rtsp://username:password@IP:8554/stream1"

        print("[+] FFMPEG: Initializing FFmpeg RTSP stream...")

        self.ffmpeg_cmd = [
            'ffmpeg',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-rtsp_transport', 'tcp',
            '-i', rtsp_url,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-'
        ]

        self.pipe = subprocess.Popen(
            self.ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

    def get_frame(self):
        last_valid_frame = None
        flush_reads = 3 

        for _ in range(flush_reads):
            raw_image = self.pipe.stdout.read(self.width * self.height * 3)
            if len(raw_image) == (self.width * self.height * 3):
                last_valid_frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((self.height, self.width, 3))

        if last_valid_frame is None:
            print("[-] Failed to grab frame")
            return False, None

        return True, last_valid_frame

    def __del__(self):
        if self.pipe:
            self.pipe.terminate()
            self.pipe.wait()
