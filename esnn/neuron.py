class Neuron:
    def __init__(self, w, t, label, merge_count=1, psp=0):
        self.w = w
        self.t = t
        self.label = label
        self.merge_count = merge_count
        self.psp = psp

    def update(self, w, t):
        self.w = (self.merge_count * self.w + w) / (1 + self.merge_count)
        self.t = (self.merge_count * self.t + t) / (1 + self.merge_count)
        self.merge_count += 1
