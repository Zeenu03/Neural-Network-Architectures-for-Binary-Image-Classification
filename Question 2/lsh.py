import numpy as np

class knn_lsh() :
    def __init__(self) :
        pass

    def train(self, data:np.ndarray, p:int) :
        n, d = data.shape
        mu = np.append(np.mean(data, axis=0), 0)
        std = np.append(np.std(data, axis=0), 1)
        w = np.random.normal(size=(p, d+1))
        a = np.ones(n)
        data = np.column_stack((data, a))
        lab = np.zeros(n)
        mat = np.matmul(data, w.transpose())
        for i in range(n) :
            for j in range(p) :
                if (mat[i][j] > 0) :
                    lab[i] += 2**j
        self.labels = lab
        self.wt = w
        self.p = p
        self.data = data[:,:-1]
    
    def predict(self, k:int, point:np.ndarray) :
        mat = np.matmul(np.append(point, 1), self.wt.transpose())
        l = 0
        for i in range(self.p) :
            if (mat[i] > 0) :
                l += 2**i
        pts = np.where(self.labels == l)
        dist = np.linalg.norm(self.data[pts] - point, axis = -1)
        knn = np.argpartition(dist, k)[:k]
        return knn