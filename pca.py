import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components    # 降维后的维度，要保留的主成分数量
        self.mean = None                    # 数据均值
        self.components = None              # 投影矩阵（主成分特征向量构成）
    
    def fit(self, X):
        # 1. 数据中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. 计算协方差矩阵
        # X及X_centered的形状: (n_samples, n_features)
        # np.cov要求输入为(n_features, n_samples)
        # 输出协方差形状: (n_features, n_features)
        cov_matrix = np.cov(X_centered.T)   # 转置并计算协方差矩阵    
        
        # 3. 特征值分解：计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. 选择主成分
        idx = eigenvalues.argsort()[::-1]   # 降序索引
        eigenvectors = eigenvectors[:, idx] # 选取所有的行，按索引排序列，即获取特征值顺序对应的特征向量
        
        # 5. 取前n_components个
        self.components = eigenvectors[:, :self.n_components]
        
        return self
    
    def transform(self, X):
        X_centered = X - self.mean
        
        # 数据投影到主成分空间
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        # 训练PCA并转换数据
        self.fit(X)
        return self.transform(X)