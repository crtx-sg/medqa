import datetime

class AppLogger:
    """A simple in-memory logger for the Streamlit app."""
    def __init__(self):
        self._logs = []

    def log(self, message: str):
        """Adds a timestamped log message."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._logs.append(f"[{timestamp}] {message}\n")

    def get_logs(self):
        """Returns all captured logs."""
        return self._logs

    def clear(self):
        """Clears all logs."""
        self._logs = []
