import asyncio
import cv2
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.cctv.camera_manager import camera_manager
from app.core.logging import logger

router = APIRouter(prefix="/ws", tags=["Real-time Streaming"])

@router.websocket("/cctv/{camera_id}")
async def websocket_cctv_stream(websocket: WebSocket, camera_id: int):
    """Streams real-time JPEG frames with bounding box, identity label, and attendance status overlays."""
    await websocket.accept()
    logger.info(f"WebSocket client connected to camera stream {camera_id}")

    try:
        while True:
            worker = camera_manager.get_worker(camera_id)
            if worker and worker.latest_annotated_frame is not None:
                frame = worker.latest_annotated_frame
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    await websocket.send_bytes(buffer.tobytes())

            await asyncio.sleep(0.06)  # ~15-20 FPS WebSocket transmission

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from camera stream {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket error on camera {camera_id}: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
