import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components    # 降维后的维度，要保留的主成分数量
        self.mean = None                    # 数据均值
        self.components = None              # 主成分（特征向量）
    
    def fit(self, X):
        """训练PCA：计算主方向"""
        # 1. 中心化（减均值）
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. 计算协方差矩阵
        # X形状: (n_samples, n_features)
        # 协方差形状: (n_features, n_features)
        cov_matrix = np.cov(X_centered.T)
        
        # 3. 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. 排序（从大到小）
        idx = eigenvalues.argsort()[::-1]  # 降序索引
        eigenvectors = eigenvectors[:, idx]
        
        # 5. 取前n_components个
        self.components = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        """降维：把数据投影到主方向上"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        """训练并转换数据（一步完成）"""
        self.fit(X)
        return self.transform(X)