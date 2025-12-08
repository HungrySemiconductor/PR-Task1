# knn_kdtree.py - 简化版KNN
import numpy as np

class KNN:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X    # 保存训练数据
        self.y_train = y    # 保存训练标签
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]  # 获取测试样本数量，行数
        predictions = np.zeros(n_samples, dtype=self.y_train.dtype) # 创建空数组，数组数据类型和y_train相同，用于存储预测结果
        
        for i in range(n_samples):
            # 1. 计算与所有训练样本的距离
            distances = np.sqrt(np.sum((self.X_train - X[i])**2, axis=1))
            
            # 2. 找最近的k个邻居
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            nearest_labels = self.y_train[nearest_indices]
            
            # 3. 投票决定类别
            unique, counts = np.unique(nearest_labels, return_counts=True)  # 统计每个标签出现的次数
            predictions[i] = unique[np.argmax(counts)]  # 找到出现次数最多的标签作为预测结果
        
        return predictions
