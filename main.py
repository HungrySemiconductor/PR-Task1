import numpy as np
import time
import json

# 导入您的自定义模块
from data_loader import load_mnist_pytorch, split_validation
from pca import PCA
from lda import LDA
from qdf import QDF_RDA
from knn import KNN

# 自己实现准确率计算函数
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def run_experiment():
    
    # 1. 加载数据
    print("\n1. 加载MNIST数据...")
    X_train, y_train, X_test, y_test = load_mnist_pytorch('./mnist_data')
    X_train, y_train, X_val, y_val = split_validation(X_train, y_train, val_ratio=0.1)
    
    # 2. 实验配置
    dimensions = [5, 9, 20, 50, 100]            # 维度
    knn_k_values = [1, 3, 5, 10]                # k值
    qdf_reg_params = [0, 0.25, 0.5, 0.75, 1]    # 正则化参数
    
    # 3. 存储结果
    results = []
    
    # 4. 运行实验
    for dim in dimensions:
        print(f"\n{'='*40}")
        print(f"降维维度: {dim}")
        print(f"{'='*40}")
        
        # PCA降维
        print(f"\n[PCA降维]")
        pca = PCA(n_components=dim)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)
        
        # QDF在PCA空间
        for reg in qdf_reg_params:
            start_time = time.time()
            qdf = QDF_RDA(reg_param=reg)
            qdf.fit(X_train_pca, y_train)
            y_val_pred = qdf.predict(X_val_pca)
            val_acc = compute_accuracy(y_val, y_val_pred)  # 使用自己的函数
            train_time = time.time() - start_time
            
            start_time = time.time()
            y_test_pred = qdf.predict(X_test_pca)
            test_acc = compute_accuracy(y_test, y_test_pred)  # 使用自己的函数
            test_time = time.time() - start_time
            
            results.append({
                '降维': 'PCA',
                '分类器': 'QDF',
                '维度': dim,
                '参数': f'reg={reg}',
                '验证准确率': val_acc,
                '测试准确率': test_acc,
                '测试时间': test_time
            })
            
            print(f"  QDF(reg={reg}): 验证={val_acc:.4f}, 测试={test_acc:.4f}, 时间={test_time:.3f}s")
        
        # KNN在PCA空间
        for k in knn_k_values:
            start_time = time.time()
            knn = KNN(n_neighbors=k)
            knn.fit(X_train_pca, y_train)
            y_val_pred = knn.predict(X_val_pca)
            val_acc = compute_accuracy(y_val, y_val_pred)  # 使用自己的函数
            train_time = time.time() - start_time
            
            start_time = time.time()
            y_test_pred = knn.predict(X_test_pca)
            test_acc = compute_accuracy(y_test, y_test_pred)  # 使用自己的函数
            test_time = time.time() - start_time
            
            results.append({
                '降维': 'PCA',
                '分类器': 'KNN',
                '维度': dim,
                '参数': f'k={k}',
                '验证准确率': val_acc,
                '测试准确率': test_acc,
                '测试时间': test_time
            })
            
            print(f"  KNN(k={k}): 验证={val_acc:.4f}, 测试={test_acc:.4f}, 时间={test_time:.3f}s")
        
        # LDA降维 (注意维度限制)
        if dim <= len(np.unique(y_train)) - 1:
            print(f"\n[LDA降维]")
            lda = LDA(n_components=dim)
            X_train_lda = lda.fit_transform(X_train, y_train)
            X_val_lda = lda.transform(X_val)
            X_test_lda = lda.transform(X_test)
            
            # QDF在LDA空间
            for reg in qdf_reg_params:
                start_time = time.time()
                qdf = QDF_RDA(reg_param=reg)
                qdf.fit(X_train_lda, y_train)
                y_val_pred = qdf.predict(X_val_lda)
                val_acc = compute_accuracy(y_val, y_val_pred)  # 使用自己的函数
                train_time = time.time() - start_time
                
                start_time = time.time()
                y_test_pred = qdf.predict(X_test_lda)
                test_acc = compute_accuracy(y_test, y_test_pred)  # 使用自己的函数
                test_time = time.time() - start_time
                
                results.append({
                    '降维': 'LDA',
                    '分类器': 'QDF',
                    '维度': dim,
                    '参数': f'reg={reg}',
                    '验证准确率': val_acc,
                    '测试准确率': test_acc,
                    '测试时间': test_time
                })
                
                print(f"  QDF(reg={reg}): 验证={val_acc:.4f}, 测试={test_acc:.4f}, 时间={test_time:.3f}s")
            
            # KNN在LDA空间
            for k in knn_k_values:
                start_time = time.time()
                knn = KNN(n_neighbors=k)
                knn.fit(X_train_lda, y_train)
                y_val_pred = knn.predict(X_val_lda)
                val_acc = compute_accuracy(y_val, y_val_pred)  # 使用自己的函数
                train_time = time.time() - start_time
                
                start_time = time.time()
                y_test_pred = knn.predict(X_test_lda)
                test_acc = compute_accuracy(y_test, y_test_pred)  # 使用自己的函数
                test_time = time.time() - start_time
                
                results.append({
                    '降维': 'LDA',
                    '分类器': 'KNN',
                    '维度': dim,
                    '参数': f'k={k}',
                    '验证准确率': val_acc,
                    '测试准确率': test_acc,
                    '测试时间': test_time
                })
                
                print(f"  KNN(k={k}): 验证={val_acc:.4f}, 测试={test_acc:.4f}, 时间={test_time:.3f}s")
    
    # 5. 输出对比表格
    print(f"\n{'='*80}")
    print("实验结果对比表")
    print(f"{'='*80}")
    print(f"{'降维方法':<8} {'分类器':<6} {'维度':<6} {'参数':<10} {'验证准确率':<12} {'测试准确率':<12} {'测试时间(s)':<12}")
    print(f"{'-'*80}")
    
    # 按测试准确率排序
    results.sort(key=lambda x: x['测试准确率'], reverse=True)
    
    for r in results:
        print(f"{r['降维']:<8} {r['分类器']:<6} {r['维度']:<6} {r['参数']:<10} "
              f"{r['验证准确率']:<12.4f} {r['测试准确率']:<12.4f} {r['测试时间']:<12.3f}")
    
    print(f"\n{'='*80}")
    print("实验完成！")
    print("可以根据上表撰写实验结果分析。")
    
    return results

if __name__ == "__main__":
    # 运行实验
    experiment_results = run_experiment()
    
    
    # 找到最佳结果
    best_by_accuracy = max(experiment_results, key=lambda x: x['测试准确率'])
    best_by_speed = min(experiment_results, key=lambda x: x['测试时间'])
    
    print(f"最高准确率组合: {best_by_accuracy['降维']}+{best_by_accuracy['分类器']}, "
          f"维度={best_by_accuracy['维度']}, 参数={best_by_accuracy['参数']}, "
          f"准确率={best_by_accuracy['测试准确率']:.4f}")
    
    print(f"最快测试速度: {best_by_speed['降维']}+{best_by_speed['分类器']}, "
          f"维度={best_by_speed['维度']}, 参数={best_by_speed['参数']}, "
          f"时间={best_by_speed['测试时间']:.3f}秒")
    
    # === 将实验结果保存为 JSON 文件 ===
    output_filename = 'experiment_results.json'
    with open(output_filename, 'w') as f:
        json.dump(experiment_results, f, indent=4)
    print(f"\n 结果已保存到 {output_filename}")