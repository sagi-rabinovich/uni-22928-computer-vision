import datetime
import logging


class MyLogger:
    PICKLE_COMPATIBILITY = True

    @staticmethod
    def getLogger(logger_name):
        if MyLogger.PICKLE_COMPATIBILITY:
            return ConsoleLogger(logger_name)
        return logging.getLogger(logger_name)


class NopLogger:
    def __init__(self, logger_name):
        self.logger_name = logger_name

    def info(self, str):
        pass


class ConsoleLogger:
    def __init__(self, logger_name):
        self.logger_name = logger_name

    def info(self, str):
        print("%s - %s - %s" % (datetime.datetime.now(), self.logger_name, str))
