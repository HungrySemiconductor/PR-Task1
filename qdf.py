# qdf_rda.py - 带RDA正则化的QDF
import numpy as np

class QDF_RDA:
    def __init__(self, reg_param=0.5):
        self.reg_param = reg_param  # 正则化参数
        self.classes = None
        self.means = None      # 各类别均值
        self.covs = None       # 正则化后的协方差矩阵
        self.priors = None     # 先验概率
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        
        # 初始化存储
        self.means = np.zeros((n_classes, n_features))
        self.covs = np.zeros((n_classes, n_features, n_features))
        self.priors = np.zeros(n_classes)
        
        for i, c in enumerate(self.classes):
            # 当前类别的样本
            X_c = X[y == c]
            n_c = len(X_c)
            
            # 先验概率
            self.priors[i] = n_c / len(X)
            
            # 均值
            self.means[i] = np.mean(X_c, axis=0)
            
            # 协方差矩阵
            X_c_centered = X_c - self.means[i]
            if n_c > 1:
                cov = np.dot(X_c_centered.T, X_c_centered) / n_c
            else:
                cov = np.zeros((n_features, n_features))
            
            # RDA正则化
            reg_cov = self._apply_rda(cov)
            self.covs[i] = reg_cov
        
        return self
    
    def _apply_rda(self, cov):
        """
        reg_param: 
        0: 标准QDF（容易出错）
        0.5: 一半用样本协方差，一半用球面协方差
        1: 完全球面协方差（稳定）
        """
        n_features = cov.shape[0]
        
        if self.reg_param == 0:
            return cov
        elif self.reg_param == 1:
            # 完全正则化：球面协方差
            trace = np.trace(cov)
            return (trace / n_features) * np.eye(n_features)
        else:
            # RDA：混合
            trace = np.trace(cov)
            identity = np.eye(n_features)
            
            # Σ_reg = (1-λ)Σ + λ*(tr(Σ)/d)*I
            reg_cov = (1 - self.reg_param) * cov + \
                     self.reg_param * (trace / n_features) * identity
            
            # 加小常数确保可逆
            reg_cov += 1e-6 * np.eye(n_features)
            return reg_cov
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # 存储每个样本对每个类别的"分数"
        scores = np.zeros((n_samples, n_classes))
        
        for i in range(n_classes):
            mean = self.means[i]
            cov = self.covs[i]
            prior = self.priors[i]
            
            # 计算协方差的逆
            cov_inv = np.linalg.pinv(cov)
            
            # 计算行列式的对数
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                logdet = -100  # 很小的数
            
            for j in range(n_samples):
                # 马氏距离: (x-μ)^T Σ^{-1} (x-μ)
                diff = X[j] - mean
                mahalanobis = np.dot(diff.T, np.dot(cov_inv, diff))
                
                # QDF判别函数: -1/2*[mahalanobis + log|Σ|] + log(prior)
                score = -0.5 * (mahalanobis + logdet) + np.log(prior + 1e-10)
                scores[j, i] = score
        
        # 选择最高分数的类别
        predictions = self.classes[np.argmax(scores, axis=1)]
        return predictions