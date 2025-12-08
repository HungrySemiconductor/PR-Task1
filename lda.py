import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None      # 投影矩阵（特征向量构成）
    
    def fit(self, X, y):
        """训练LDA：找到最好分开各类的方向"""
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # LDA最多降到(类别数-1)维
        max_dim = min(n_classes - 1, n_features)
        if self.n_components > max_dim:
            self.n_components = max_dim
        
        # 1. 总体均值
        self.mean = np.mean(X, axis=0)
        
        # 2. 计算类内散度(S_w)和类间散度(S_b)
        S_w = np.zeros((n_features, n_features))
        S_b = np.zeros((n_features, n_features))
        
        for c in classes:
            # 当前类别的样本
            X_c = X[y == c]
            n_c = len(X_c)
            mean_c = np.mean(X_c, axis=0)
            
            # 类内散度
            X_c_centered = X_c - mean_c
            S_w += np.dot(X_c_centered.T, X_c_centered)
            
            # 类间散度
            mean_diff = (mean_c - self.mean).reshape(-1, 1) # 为计算外积，需转换为列向量
            S_b += n_c * np.dot(mean_diff, mean_diff.T)
        
        # 3. 解广义特征值问题：S_w^{-1} S_b
        S_w_inv = np.linalg.pinv(S_w)  
        matrix = np.dot(S_w_inv, S_b)
        
        # 4. 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        # 5. 排序并选择
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components].real
        
        return self
    
    def transform(self, X):
        """降维"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X, y):
        """训练并转换数据（一步完成）"""
        self.fit(X, y)
        return self.transform(X)