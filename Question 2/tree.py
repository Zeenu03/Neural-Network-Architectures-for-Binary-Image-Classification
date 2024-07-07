import numpy as np

class Tree() :
    def __init__(self, max_nodes:int):
        self.arr = np.zeros((max_nodes, 2))
        self.isleaf = np.zeros(max_nodes)

    def add_node(self, idx, axis, threshold) :
        self.arr[idx] = axis, threshold