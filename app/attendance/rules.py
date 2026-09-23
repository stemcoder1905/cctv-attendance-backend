from datetime import datetime, time
from app.core.config import settings

def parse_time_str(time_str: str) -> time:
    """Parses HH:MM time string into datetime.time object."""
    parts = time_str.split(":")
    return time(hour=int(parts[0]), minute=int(parts[1]))

def evaluate_attendance_status(first_seen_dt: datetime, late_after_time_str: str = None) -> str:
    """
    Evaluates attendance status based on first-seen time of the day.
    - If first_seen_time <= late_after_time -> PRESENT
    - If first_seen_time > late_after_time  -> LATE
    """
    if late_after_time_str is None:
        late_after_time_str = settings.LATE_AFTER_TIME

    late_time = parse_time_str(late_after_time_str)
    seen_time = first_seen_dt.time()

    if seen_time <= late_time:
        return "PRESENT"
    else:
        return "LATE"
