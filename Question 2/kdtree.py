import tree
import numpy as np

class knn_kdtree() :
    def __init__(self) :
        pass

    def train(self, X_train:np.ndarray, s:int) :
        self.arr = X_train

        n, d = (self.arr).shape
        labels = np.zeros(n)

        t = tree.Tree(n)

        def add_node(sample_idx, node_idx) :
            if (len(sample_idx) < s) :
                labels[sample_idx] = node_idx
                t.isleaf[node_idx] = 1
                return None
            
            ax = np.random.randint(d)
            threshold = np.median(self.arr[sample_idx,ax])
            t.add_node(node_idx, ax, threshold)

            add_node(sample_idx[np.where(self.arr[sample_idx,ax] < threshold)], 2*node_idx+1)
            add_node(sample_idx[np.where(self.arr[sample_idx,ax] >= threshold)], 2*node_idx+2)
        
        add_node(np.arange(n), 0)
        
        self.tree = t
        self.labels = labels
    
    def predict(self, k:int, point:np.ndarray) -> np.ndarray:
        i = 0
        while self.tree.isleaf[i] == 0 :
            if (point[int(self.tree.arr[i][0])] < self.tree.arr[i][1]) :
                i = 2*i+1
            else :
                i = 2*i+2
        X_ind = np.where(self.labels == i)[0]
        dist = np.linalg.norm(self.arr[X_ind]-point, axis=-1)
        a = np.array(np.argpartition(dist, k)[:k])
        knn = X_ind[a]
        return knn