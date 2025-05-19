
import asyncio
import threading
import websockets
import base64
import cv2
import numpy as np

class VideoCamera:
    def __init__(self, uri="ws://IP:PORT", ping_interval=40):
        self.uri = uri
        self.frame = None
        self.loop = asyncio.new_event_loop()
        self.ping_interval = ping_interval  
        self.ping_task = None

        self.thread = threading.Thread(target=self._start_ws_client, daemon=True)
        self.thread.start()

    def _start_ws_client(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._receive_frames())

    async def _receive_frames(self):
        while True:
            try:
                async with websockets.connect(self.uri, ping_interval=self.ping_interval) as websocket:
                    print("[+] Connected to WebSocket server")
                    
                    self.ping_task = self.loop.create_task(self._send_pings(websocket))

                    async for data in websocket:
                        jpg_original = base64.b64decode(data)
                        jpg_np = np.frombuffer(jpg_original, dtype=np.uint8)
                        img = cv2.imdecode(jpg_np, cv2.IMREAD_COLOR)
                        if img is not None:
                            self.frame = img
            except Exception as e:
                print("[-] WebSocket error:", e)
                await asyncio.sleep(5)  
            finally:
                if self.ping_task:
                    self.ping_task.cancel()

    async def _send_pings(self, websocket):
        while True:
            try:
                print("[+] Pinged")
                await websocket.ping()  
                await asyncio.sleep(self.ping_interval) 
            except Exception as e:
                print("[-] Ping error:", e)
                break

    def get_frame(self):
        if self.frame is not None:
            return True, self.frame.copy()
        else:
            return False, None

    async def close(self):
        print("[*] Closing VideoCamera...")
        if self.ping_task:
            self.ping_task.cancel()
        self.loop.stop()
        await asyncio.sleep(0) 

    def restart(self):
        asyncio.run(self.close())  
        self.__init__(self.uri, self.ping_interval) 
