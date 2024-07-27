# TODO:
# - Make indentation start with message section

import logging
import time
from utils.config_manager import ConfigManager
from threading import Lock
import textwrap

# Add custom logging levels
MODEL = 26  # Between INFO (20) and WARNING (30)
TRAIN = 27  # Between MODEL (25) and WARNING (30)
INFER = 28  # Between TRAIN (26) and WARNING (30)

logging.addLevelName(MODEL, "MODEL")
logging.addLevelName(TRAIN, "TRAIN")
logging.addLevelName(INFER, "INFER")

class ElapsedTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None, model_fmt=None, train_fmt=None, infer_fmt=None, datefmt=None, max_width=120):
        self.fmt = fmt
        self.model_fmt = model_fmt
        self.train_fmt = train_fmt
        self.infer_fmt = infer_fmt
        self.datefmt = datefmt
        self.start_time = time.perf_counter()
        self.lock = Lock()
        self.max_width = max_width

    def format(self, record):
        with self.lock:
            elapsed = time.perf_counter() - self.start_time
            record.elapsed_time = f"{elapsed:.6f}s"
            
            if record.levelno == logging.INFO:
                # Apply dynamic borders for INFO level
                formatted = logging.Formatter(self.fmt, self.datefmt).format(record)
                max_content_width = self.max_width - 6  # -6 for side padding and dashes
                wrapped_lines = textwrap.wrap(formatted, width=max_content_width)
                
                max_line_length = max(len(line) for line in wrapped_lines)
                max_line_length = max(max_line_length, 130)  # Ensure minimum content width of 24
                total_width = max_line_length + 6  # +6 for side padding and dashes
                
                border = "-" * total_width
                
                final_message = f"{border}\n"
                for line in wrapped_lines:
                    left_padding = (total_width - len(line) - 2) // 2
                    right_padding = total_width - len(line) - 2 - left_padding
                    final_message += f"{'-' * left_padding} {line} {'-' * right_padding}\n"
                final_message += border
                
                return final_message
            elif record.levelno == MODEL:
                formatted = logging.Formatter(self.model_fmt, self.datefmt).format(record)
                wrapped_lines = textwrap.wrap(formatted, width=self.max_width - 4)
                return "    " + "\n    ".join(wrapped_lines)
            elif record.levelno == TRAIN:
                formatted = logging.Formatter(self.train_fmt, self.datefmt).format(record)
                wrapped_lines = textwrap.wrap(formatted, width=self.max_width - 4)
                return "    " + "\n    ".join(wrapped_lines)
            elif record.levelno == INFER:
                formatted = logging.Formatter(self.infer_fmt, self.datefmt).format(record)
                wrapped_lines = textwrap.wrap(formatted, width=self.max_width - 4)
                return "    " + "\n    ".join(wrapped_lines)
            else:
                return logging.Formatter(self.fmt, self.datefmt).format(record)
            
def setup_logging():
    config = ConfigManager()
    
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = ElapsedTimeFormatter(
        fmt=config.get('LOG_FORMAT'),
        model_fmt=config.get('MODEL_LOG_FORMAT'),
        train_fmt=config.get('TRAIN_LOG_FORMAT'),
        infer_fmt=config.get('INFER_LOG_FORMAT')
    )
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)

    # Add custom methods to both Logger class and logging module
    def model(self, message, *args, **kwargs):
        if isinstance(self, logging.Logger):
            self.log(MODEL, message, *args, **kwargs)
        else:
            logging.log(MODEL, message, *args, **kwargs)

    def train(self, message, *args, **kwargs):
        if isinstance(self, logging.Logger):
            self.log(TRAIN, message, *args, **kwargs)
        else:
            logging.log(TRAIN, message, *args, **kwargs)

    def infer(self, message, *args, **kwargs):
        if isinstance(self, logging.Logger):
            self.log(INFER, message, *args, **kwargs)
        else:
            logging.log(INFER, message, *args, **kwargs)

    logging.Logger.model = model
    logging.Logger.train = train
    logging.Logger.infer = infer

    logging.model = lambda message, *args, **kwargs: logging.log(MODEL, message, *args, **kwargs)
    logging.train = lambda message, *args, **kwargs: logging.log(TRAIN, message, *args, **kwargs)
    logging.infer = lambda message, *args, **kwargs: logging.log(INFER, message, *args, **kwargs)

# Add this line at the end of the file
setup_logging()