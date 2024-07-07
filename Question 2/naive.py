import numpy as np

class Naive() :
    def __init__(self):
        pass
    def train(self, X_train:np.ndarray) :
        self.arr = np.array(X_train)
    
    def predict(self, k:int, point:np.ndarray) :
        self.k = k
        dist = np.linalg.norm(self.arr - point, axis = -1)
        knn = np.argpartition(dist, self.k)[:self.k]
        return knn