import cv2
import time
import threading
import numpy as np
from typing import Optional, Tuple, Union
from app.core.logging import logger
from app.core.security import sanitize_camera_url

class CameraStreamReader:
    """Non-blocking threaded OpenCV stream reader with auto-reconnection backoff."""

    def __init__(self, camera_id: int, stream_url: Union[str, int], camera_name: str = "Camera"):
        self.camera_id = camera_id
        self.raw_stream_url = stream_url
        self.camera_name = camera_name
        self.sanitized_url = sanitize_camera_url(str(stream_url))
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.status: str = "OFFLINE"  # ONLINE, DISCONNECTED, ERROR, OFFLINE
        
        self.latest_frame: Optional[np.ndarray] = None
        self.last_frame_timestamp: float = 0.0
        
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        
        # Reconnection parameters
        self.reconnect_interval: float = 5.0
        self.max_reconnect_attempts: int = 10

    def start(self):
        """Starts background frame reader thread."""
        if not self.is_running:
            self.is_running = True
            self._thread = threading.Thread(target=self._read_loop, name=f"CameraReader-{self.camera_id}", daemon=True)
            self._thread.start()
            logger.info(f"Started camera stream reader for [{self.camera_name}] (URL: {self.sanitized_url}).")

    def stop(self):
        """Stops stream reader and releases video capture object."""
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release_cap()
        self.status = "OFFLINE"
        logger.info(f"Stopped camera stream reader for [{self.camera_name}].")

    def _open_stream(self) -> bool:
        """Opens video capture connection."""
        self._release_cap()
        try:
            # Parse integer for webcam index (e.g. "0" -> 0)
            target = self.raw_stream_url
            if isinstance(target, str) and target.isdigit():
                target = int(target)

            logger.info(f"Connecting camera [{self.camera_name}] ({self.sanitized_url})...")
            self.cap = cv2.VideoCapture(target)
            
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self._lock:
                        self.latest_frame = frame
                        self.last_frame_timestamp = time.time()
                        self.status = "ONLINE"
                    logger.info(f"Camera [{self.camera_name}] connected successfully. Resolution: {frame.shape[1]}x{frame.shape[0]}")
                    return True
            
            self.status = "DISCONNECTED"
            return False
        except Exception as e:
            logger.error(f"Failed to open stream for [{self.camera_name}]: {e}")
            self.status = "ERROR"
            return False

    def _read_loop(self):
        """Thread reading loop draining buffer continuously."""
        attempts = 0
        while self.is_running:
            if self.cap is None or not self.cap.isOpened() or self.status != "ONLINE":
                attempts += 1
                connected = self._open_stream()
                if not connected:
                    time.sleep(min(self.reconnect_interval * attempts, 30.0))
                    continue
                else:
                    attempts = 0

            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning(f"Stream dropped for camera [{self.camera_name}]. Will attempt reconnection...")
                self.status = "DISCONNECTED"
                self._release_cap()
                time.sleep(self.reconnect_interval)
                continue

            with self._lock:
                self.latest_frame = frame
                self.last_frame_timestamp = time.time()
                self.status = "ONLINE"

            # Sleep slightly to avoid 100% CPU thread spin on synthetic streams
            time.sleep(0.01)

    def read_latest_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """Returns the latest buffered frame safely."""
        with self._lock:
            if self.latest_frame is not None and self.status == "ONLINE":
                return True, self.latest_frame.copy(), self.last_frame_timestamp
            return False, None, 0.0

    def _release_cap(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
