#!/bin/bash

# Check if mediamtx is in PATH
if which mediamtx > /dev/null 2>&1; then
    echo "[+] Mediamtx is installed at: $(which mediamtx)"
else
    echo "[*] Installing Mediamtx"
    wget https://github.com/bluenviron/mediamtx/releases/download/v1.12.0/mediamtx_v1.12.0_linux_armv7.tar.gz
    tar -xvzf mediamtx_v1.12.0_linux_armv7.tar.gz
    cd mediamtx
    sudo mv mediamtx /usr/local/bin/
    sudo chmod +x /usr/local/bin/mediamtx
    
    if which ffmpeg > /dev/null 2>&1; then
        echo echo "[+] Mediamtx is installed at: $(which mediamtx)"
    else
        echo "[-] Mediamtx installation failed."
    fi

fi



if which ffmpeg > /dev/null 2>&1; then
    echo "[+] FFmpeg is already installed: $(ffmpeg -version | head -n 1)"
else
    echo "[*] FFmpeg not found. Installing"
    sudo apt update
    sudo apt install ffmpeg -y
    if which ffmpeg > /dev/null 2>&1; then
        echo "[+] FFmpeg installed successfully."
    else
        echo "[-] FFmpeg installation failed."
    fi
fi

sed -i "s|ExecStart=.*|ExecStart=/bin/bash $(pwd)/start_cam.sh|" /etc/systemd/system/start_cam.service

sudo chmod +x start_cam.sh
sudo chmod +x enable
sudo chmod +x disable
./enable
