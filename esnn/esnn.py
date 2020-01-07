from collections import defaultdict

class ESNN():
    def __init__(self, m=0.9, c=0.7, s=0.6):
        self.repository = defaultdict(list)
        self.m = m
        self.c = c
        self.s = s
