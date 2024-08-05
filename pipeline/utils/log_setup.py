import logging
import time
from utils.config_manager import ConfigManager
from threading import Lock
import textwrap

# Add custom PIPELINE level
PIPELINE = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(PIPELINE, "PIPELINE")

class ElapsedTimeFormatter(logging.Formatter):
    def __init__(self, fmt=None, pipeline_fmt=None, datefmt=None, max_width=120):
        self.fmt = fmt
        self.pipeline_fmt = pipeline_fmt
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
            elif record.levelno in [PIPELINE, logging.WARNING]:
                formatted = logging.Formatter(self.pipeline_fmt, self.datefmt).format(record)
                # Wrap the message for PIPELINE and WARNING levels
                wrapped_lines = textwrap.wrap(formatted, width=self.max_width - 4)  # -4 for initial indentation
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
        pipeline_fmt=config.get('PIPELINE_LOG_FORMAT')
    )
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(ch)

    # Add pipeline method to both Logger class and logging module
    def pipeline(self, message, *args, **kwargs):
        if isinstance(self, logging.Logger):
            self.log(PIPELINE, message, *args, **kwargs)
        else:
            logging.log(PIPELINE, message, *args, **kwargs)

    logging.Logger.pipeline = pipeline
    logging.pipeline = lambda message, *args, **kwargs: logging.log(PIPELINE, message, *args, **kwargs)

# Add this line at the end of the file
setup_logging()