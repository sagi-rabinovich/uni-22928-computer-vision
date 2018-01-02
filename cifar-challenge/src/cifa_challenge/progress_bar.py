# coding=utf-8
import datetime


class ProgressBar:
    def __init__(self):
        """
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)

        """
        self.prefix = 'Progress: '
        self.suffix = ''
        self.decimals = 1
        self.length = 100
        self.fill = '>'
        self.iteration = 0
        self.total = 0
        self._start = None
        self._end = None
        self._update_after_n_intervals = 1

    def track(self, items):
        total = len(items)
        self.start(total)
        for item in items:
            yield item
            self.increment()

    def start(self, total):
        print
        self.iteration = 0
        self.total = total
        self._start = datetime.datetime.now()
        self._update_after_n_intervals = max(total / 1000, 1)
        self.update(0, total)

    # Print iterations progress
    def update(self, iteration, total):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
        """
        if iteration % self._update_after_n_intervals:
            return

        completed = (iteration / float(total))
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * completed)
        filledLength = int(self.length * iteration // total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        eta = datetime.timedelta(
            seconds=(datetime.datetime.now() - self._start).total_seconds() / completed * (
                    1 - completed)) if completed > 0 else ''
        print '\r%s |%s| %s%% %s [ETA: %s]' % (self.prefix, bar, percent, self.suffix, eta),
        # Print New Line on Complete
        if iteration == total:
            print

    def increment(self):
        self.iteration += 1
        self.update(self.iteration, self.total)

    def done(self):
        self.iteration = self.total
        self.update(self.iteration, self.total)
