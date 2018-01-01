# coding=utf-8
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
        self.fill = '█'
        self.iteration = 0
        self.total = 0

    def forEach(self, items, action):
        total = len(items)
        self.start(total)
        for item in items:
            action(item)
            self.increment()

    def start(self, total):
        print
        self.iteration = 0
        self.total = total
        self.update(0, total)

    # Print iterations progress
    def update(self, iteration, total):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
        """
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(self.length * iteration // total)
        bar = self.fill * filledLength + '-' * (self.length - filledLength)
        print '\r%s |%s| %s%% %s' % (self.prefix, bar, percent, self.suffix),
        # Print New Line on Complete
        if iteration == total:
            print

    def increment(self):
        self.iteration += 1
        self.update(self.iteration, self.total)

    def done(self):
        self.iteration = self.total
        self.update(self.iteration, self.total)
