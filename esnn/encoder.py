import numpy as np

class Encoder():
    def __init__(self, num_fields, beta, i_min, i_max):
        self.num_fields = num_fields
        self.beta = beta
        self.i_min = i_min
        self.i_max = i_max
        self.fields = np.arange(num_fields)
        self.mu = self.i_min + (2 * self.fields - 3) * (self.i_max - self.i_min) / (2. * (self.num_fields - 2))
        self.sigma = (self.i_max - self.i_min) / (self.beta * (self.num_fields - 2))

        def rcf(x):
            return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(- np.square(x - self.mu) / (2. * self.sigma**2))

        self.rcf = np.vectorize(rcf)

    def encode(self, vec):
        vec = vec.flatten()
        res = np.zeros([vec.shape[0], self.num_fields])
        for index, val in np.ndenumerate(vec):
            res[index[0]:] = self.rcf(val)
        return res
