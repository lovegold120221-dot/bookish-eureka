from eburon.helpers.logger_config import configure_logger

logger = configure_logger(__name__)


class BaseManager:
    def __init__(self):
        self.agent = "eburon-agent"
