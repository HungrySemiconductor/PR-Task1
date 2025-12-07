# knn_kdtree.py - 简化版KNN
import numpy as np

class SimpleKNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """训练：只需要保存数据"""
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """预测"""
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=self.y_train.dtype)
        
        for i in range(n_samples):
            # 1. 计算与所有训练样本的距离
            distances = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
            
            # 2. 找最近的k个邻居
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            
            # 3. 投票决定类别
            unique, counts = np.unique(nearest_labels, return_counts=True)
            predictions[i] = unique[np.argmax(counts)]
        
        return predictions

