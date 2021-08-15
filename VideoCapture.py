import cv2
import queue
import threading

class VideoCapture:

    def __init__(self):
        self.cap = cv2.VideoCapture(-1, cv2.CAP_V4L)
        self.q = queue.Queue()
        t = threading.Thread(target = self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()

            if not ret:
                break

            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def release(self):
        self.cap.release()