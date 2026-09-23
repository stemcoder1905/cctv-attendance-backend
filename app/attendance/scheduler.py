import asyncio
from datetime import datetime, timezone, timedelta
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.attendance.engine import process_cutoff_absent_records
from app.core.logging import logger

class AttendanceScheduler:
    """Background scheduler managing cutoff ABSENT generation and data retention cleanup tasks."""

    def __init__(self):
        self._task: asyncio.Task = None
        self._running: bool = False

    def start(self):
        """Starts background loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._scheduler_loop())
            logger.info("Attendance background scheduler started.")

    def stop(self):
        """Stops background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            logger.info("Attendance background scheduler stopped.")

    async def _scheduler_loop(self):
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every 60 seconds
                now = datetime.now(timezone.utc)
                now_time_str = now.strftime("%H:%M")

                # Check if current time matches cutoff time (e.g. 17:00)
                if now_time_str == settings.ATTENDANCE_CUTOFF_TIME:
                    logger.info(f"Cutoff time ({settings.ATTENDANCE_CUTOFF_TIME}) reached. Triggering ABSENT records generation...")
                    async with AsyncSessionLocal() as db:
                        await process_cutoff_absent_records(db, target_date=now.date())
                    # Sleep 65s to prevent multi-trigger within same minute
                    await asyncio.sleep(65)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background scheduler loop: {e}")

attendance_scheduler = AttendanceScheduler()
