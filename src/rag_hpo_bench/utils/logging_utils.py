import logging


class ModuleNameFormatter(logging.Formatter):
    """Custom formatter that shows only the module name, not the full package path."""
    
    def format(self, record):
        # Extract only the module name from the full package path
        record.module_only = record.name.split('.')[-1]
        return super().format(record)


def init_logger(level=logging.INFO):
    """
    Initialize logging with custom format.
    
    Format: YYYY-MM-DD HH:MM:SS,mmm [LEVEL]  module_name: message
    Example: 2026-02-25 11:19:05,790 [INFO]  tune_and_test_runner: Running tuner ..
    
    Args:
        level: Logging level (default: logging.INFO)
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ModuleNameFormatter(
        fmt="%(asctime)s [%(levelname)s]  %(module_only)s: %(message)s"
    ))
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicate logs
    root_logger.handlers.clear()
    
    root_logger.addHandler(handler)