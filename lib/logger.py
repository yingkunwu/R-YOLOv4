import logging
from colorlog import ColoredFormatter
from torch.utils.tensorboard import SummaryWriter

# this logger is for tensorboard
class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def list_of_scalars_summary(self, tag_value_dictionary, step):
        """Log scalar variables."""
        for tag, value in tag_value_dictionary.items():
            self.writer.add_scalar(tag, value, global_step=step)

# this logger is for logging formatter
def setup_logger(log_file_path: str = None):
    """Return a logger with a default ColoredFormatter."""
    formatter = ColoredFormatter(
        "%(asctime)s %(log_color)s%(levelname)-8s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        reset=True,
        log_colors={
            'DEBUG': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red',
        })

    logger = logging.getLogger(__name__)
    shandler = logging.StreamHandler()
    shandler.setFormatter(formatter)
    shandler.setLevel(level=logging.INFO)

    logger.addHandler(shandler)
    logger.setLevel(level=logging.INFO)
    return logger

logger = setup_logger()
