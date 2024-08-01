import os

import structlog


class ConditionalLogger:
    def __init__(self, should_log: bool):
        self.logger = structlog.get_logger()
        self.should_log = should_log

    def debug(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.debug(msg, *posargs, **kwargs)

    def info(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.info(msg, *posargs, **kwargs)

    def warning(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.warning(msg, *posargs, **kwargs)

    def error(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.error(msg, *posargs, **kwargs)

    def fatal(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.fatal(msg, *posargs, **kwargs)

    def exception(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.exception(msg, *posargs, **kwargs)

    def critical(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.critical(msg, *posargs, **kwargs)

    def msg(self, msg: str, *posargs, **kwargs) -> None:
        if self.should_log:
            self.logger.msg(msg, *posargs, **kwargs)


class MultiProcessLogger(ConditionalLogger):
    def __init__(self):
        super().__init__(should_log=is_main_process())


def is_multiprocessing() -> bool:
    # Both torch DDP and deepspeed set this env variable to the local rank of the process.
    # If the variable is not set, then we conclude that we're not in  multiprocessing environment,
    # and hence the process is the main (and only) process.
    # If we switch to a different framework or to multihost training, this method should be adapted.
    return os.environ.get("LOCAL_RANK") is not None


def get_process_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    return get_process_local_rank() == 0


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))
