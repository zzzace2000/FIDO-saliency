import time


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        print('Start %s...' % self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('Finish {} in {} seconds'.format(self.name, time.time() - self.start_time))