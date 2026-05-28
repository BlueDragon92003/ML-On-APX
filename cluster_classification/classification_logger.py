import logging

logging.basicConfig(filename="./logs/classification-latest.log", level=logging.INFO)


class ClassificationLogger:
    """A basic extention to the default python logger."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log_trace(self, msg: str):
        self.logger.log(5, msg)

    def log_debug(self, msg: str):
        self.logger.log(10, msg)

    def log_info(self, msg: str):
        self.logger.log(20, msg)

    def log_warn(self, msg: str):
        self.logger.log(30, msg)

    def log_error(self, msg: str):
        self.logger.log(40, msg)

    def log_critical(self, msg: str):
        self.logger.log(45, msg)

    def log_fatal(self, msg: str):
        self.logger.log(50, msg)
