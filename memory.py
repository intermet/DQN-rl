import numpy as np

class Memory(object):

    def __init__(self, state_shape, max_size):
        self.max_size = max_size
        self.idx = 0
        self.data = np.zeros((max_size, state_shape*2 + 3), dtype=object)
        self.is_ready = False

    def add(self, sample):
        if self.idx == self.max_size:
            self.is_ready = True
            self.idx = 0
        self.data[self.idx] = np.hstack(sample)
        self.idx += 1

    def sample(self, batch_size):
        return np.random.choice(self.max_size, batch_size)
    
    def fill(self, mem):
        self.data = mem.data.copy()

