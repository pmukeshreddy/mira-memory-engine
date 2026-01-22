"""
Application uptime tracking.
"""

import time

# Application start time for uptime tracking
APP_START_TIME: float = 0.0


def set_start_time() -> None:
    """Set the application start time."""
    global APP_START_TIME
    APP_START_TIME = time.time()


def get_uptime() -> float:
    """Get application uptime in seconds."""
    if APP_START_TIME == 0:
        return 0.0
    return time.time() - APP_START_TIME
