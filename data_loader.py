import torch
from torchvision import datasets, transforms
import numpy as np

def load_mnist_pytorch(data_dir='./mnist_data'):
    print(f"开始加载MNIST数据到{data_dir}")

    # ## 定义预处理操作
    # # 将图像转换为PyTorch张量，并自动归一化
    # # 把图像维度从(H, W)变成(C, H, W)，其中C是通道数，C=1表示灰度图
    # transform = transforms.Compose([
    #     transforms.ToTensor()    
    # ])
    
    try:
        # 加载训练集
        train_dataset = datasets.MNIST(
            root=data_dir,          # 数据存储到指定目录
            train=True,             # True为训练集，False为测试集
            transform=None,         # 不需要应用上面定义的预处理
            download=True           # 如果数据不存在则从网上下载（第一次运行）   
        )
        
        # 加载测试集
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            transform=None,
            download=True
        )
        
    except Exception as e:
        print(f"加载失败: {e}")
        print("请检查网络连接或手动下载数据")
        return None
    
    ## 数据转换和展平
    # 直接取原始数据train_dataset.data，将PyTorch张量转换为NumPy数组
    # 将28x28的二维向量展平为784的一维向量
    X_train = train_dataset.data.numpy().reshape(-1, 784).astype(np.float32)
    y_train = train_dataset.targets.numpy()     # 训练标签处理，获取图片对应的标签(0-9)
    
    X_test = test_dataset.data.numpy().reshape(-1, 784).astype(np.float32)
    y_test = test_dataset.targets.numpy()
    
    # 归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    
    print("=" * 50)
    print("数据加载完成!")
    print(f"训练集形状: {X_train.shape}")
    print(f"训练标签形状: {y_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    print(f"测试标签形状: {y_test.shape}")
    print(f"类别: {np.unique(y_train)}")
    print(f"像素值范围: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print("=" * 50)
    
    return X_train, y_train, X_test, y_test

def split_validation(X_train, y_train, val_ratio=0.1):
    # 从训练集中划分验证集，验证集比例默认为10%
    n_val = int(len(X_train) * val_ratio)
    
    # 随机打乱并划分训练集和验证集
    indices = np.random.permutation(len(X_train))   # 随机打乱训练集索引
    train_idx, val_idx = indices[n_val:], indices[:n_val]   # 划分训练集和验证集索引
    
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train_new = X_train[train_idx]
    y_train_new = y_train[train_idx]
    
    return X_train_new, y_train_new, X_val, y_val


# 使用示例
if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test = load_mnist_pytorch('./mnist_data')
    
    if X_train is not None:
        # 划分训练集与验证集
        X_train, y_train, X_val, y_val = split_validation(X_train, y_train)
        print("\n数据集划分:")
        print(f"训练集: {X_train.shape}, {y_train.shape}")
        print(f"验证集: {X_val.shape}, {y_val.shape}")
        print(f"测试集: {X_test.shape}, {y_test.shape}")
        
        # 可视化几个样本
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        # for i in range(10):
        #     ax = axes[i//5, i%5]
        #     img = X_train[i].reshape(28, 28)
        #     ax.imshow(img, cmap='gray')
        #     ax.set_title(f'Label: {y_train[i]}')
        #     ax.axis('off')
        # plt.tight_layout()
        # plt.show()