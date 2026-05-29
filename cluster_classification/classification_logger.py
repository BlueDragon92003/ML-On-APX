import logging
import os

logging.basicConfig(filename="./logs/classification-latest.log", level=logging.INFO)
logging.addLevelName(5, 'TRACE')
logging.addLevelName(45, 'CRITICAL')

class ClassificationLogger:
    """A basic extention to the default python logger."""

    def __init__(self, location):
        self.logger = logging.getLogger(__name__)
        self.location = location
        match os.environ.get('PY_LOG_LEVEL'):
            case 'TRACE':
                self.logger.setLevel( 5 )
            case 'DEBUG':
                self.logger.setLevel(10)
            case 'INGO':
                self.logger.setLevel(20)
            case 'WARN':
                self.logger.setLevel(30)
            case 'ERROR':
                self.logger.setLevel(40)
            case 'CRITICAL':
                self.logger.setLevel(45)
            case 'FATAL':
                self.logger.setLevel(50)
            case _:
                self.logger.setLevel(20)

    def log_trace(self, msg: str):
        self.logger.log(5, f"{msg} # from {self.location}")

    def log_debug(self, msg: str):
        self.logger.log(10, f"{msg} # from {self.location}")

    def log_info(self, msg: str):
        self.logger.log(20, f"{msg} # from {self.location}")

    def log_warn(self, msg: str):
        self.logger.log(30, f"{msg} # from {self.location}")

    def log_error(self, msg: str):
        self.logger.log(40, f"{msg} # from {self.location}")

    def log_critical(self, msg: str):
        self.logger.log(45, f"{msg} # from {self.location}")

    def log_fatal(self, msg: str):
        self.logger.log(50, f"{msg} # from {self.location}")
