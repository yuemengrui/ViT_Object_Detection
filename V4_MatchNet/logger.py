# *_*coding:utf-8 *_*
# @Author : yuemengrui
import sys
import time


class Log:

    def __init__(self, status=""):
        self.levels = {0: 'ERROR', 1: 'WARNING', 2: 'INFO', 3: 'DEBUG'}
        self.status = status

    def debug(self, message=""):
        self._log(level=3, message=message)

    def info(self, message=""):
        self._log(level=2, message=message)

    def warning(self, message=""):
        self._log(level=1, message=message)

    def error(self, message=""):
        self._log(level=0, message=message)

    def _log(self, level=2, message=""):
        current_time = time.time()
        time_array = time.localtime(current_time)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
        if self.status != "":
            print("[{}][{}] {}".format(current_time, self.status, message).encode(
                "utf-8").decode("latin1"))
        else:
            print("[{}] {}".format(current_time, message).encode("utf-8").decode("latin1"))
        sys.stdout.flush()


basic = Log()
train = Log('TRAIN')
val = Log('EVAL')
