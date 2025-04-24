mediamtx

sleep(10)
# Change RTSP credentials
rpicam-vid -t 0 --inline --width 640 --height 480 --framerate 15 -o - | ffmpeg -loglevel error -f h264 -i - -vcodec libx264 -preset ultrafast -tune zerolatency -b:v 3000k -maxrate 3500k -bufsize 1000k -g 30 -pix_fmt yuv420p -f rtsp -rtsp_transport udp rtsp://user:pass123@localhost:8554/stream1
