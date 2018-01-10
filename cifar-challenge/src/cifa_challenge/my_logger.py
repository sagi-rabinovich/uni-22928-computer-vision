import logging


class MyLogger:
    PICKLE_COMPATABILITY = False

    @staticmethod
    def getLogger(logger_name):
        if MyLogger.PICKLE_COMPATABILITY:
            return NopLogger()
        return logging.getLogger(logger_name)


class NopLogger:
    def info(self, str):
        pass
