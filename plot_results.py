import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  

DEFAULT_REG = 0.5
DEFAULT_K = 5

# ==========================================================
# 1. 配置与数据加载
# ==========================================================

try:
    with open('experiment_results.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("错误：未找到 experiment_results.json 文件。请先运行 main.py 生成数据。")
    exit()

# 辅助函数：提取特定条件下的指标
def get_data_point(data_list, dim, classifier, param_type, param_value, method='PCA', metric='测试准确率'):
    """筛选特定维度、分类器和参数下的指标"""
    
    param_str = f'{param_type}={param_value}' if param_type else ''
    
    filtered = [r[metric] for r in data_list 
                if r.get('降维') == method and 
                   r.get('维度') == dim and 
                   r.get('分类器') == classifier and 
                   r.get('参数') == param_str]
    
    return filtered[0] if filtered else 0.0

# 辅助函数：按降维方法和维度分组
def group_data_by_method(data_list):
    groups = {'PCA': {}, 'LDA': {}}
    for r in data_list:
        method = r['降维']
        dim = r['维度']
        if dim not in groups[method]:
            groups[method][dim] = []
        groups[method][dim].append(r)
    return groups

grouped_data = group_data_by_method(data)
pca_dims = sorted(list(grouped_data['PCA'].keys()))
lda_dims = sorted(list(grouped_data['LDA'].keys()))


# ==========================================================
# 2. 绘图函数
# ==========================================================

# ----------------------------------------------------------
# 图 4.2.1.1: QDF 正则化与模型稳定性分析
# ----------------------------------------------------------
def plot_qdf_regularization(data_list, dim, method='LDA', filename='figure_4_2_1_1.png'):
    
    # 提取子空间下的所有 QDF 正则化结果
    qdf_filtered = [r for r in data_list if r['降维'] == method and r['维度'] == dim and r['分类器'] == 'QDF']
    qdf_filtered.sort(key=lambda x: float(x['参数'].split('=')[1]))
    
    reg_params = [float(r['参数'].split('=')[1]) for r in qdf_filtered]
    val_accs = [r['验证准确率'] for r in qdf_filtered]
    test_accs = [r['测试准确率'] for r in qdf_filtered]

    plt.figure(figsize=(8, 5))
    plt.plot(reg_params, val_accs, marker='o', linestyle='-', color='red', label='验证集准确率')
    plt.plot(reg_params, test_accs, marker='x', linestyle='--', color='blue', label='测试集准确率')
    
    # --- Y轴范围 (保持从 0.5 开始) ---
    plt.ylim(0.8, 1.0) 
    
    # === 新增：添加测试集数据点标签 ===
    for reg, acc in zip(reg_params, test_accs):
        plt.text(reg, acc + 0.005, f'{acc:.2%}', ha='center', va='bottom', fontsize=9, color='blue')
    # ===================================
    
    plt.title(f'图 4.2.1.1. QDF 正则化参数 $\lambda$ 对准确率的影响 ({method} D={dim})')
    plt.xlabel('正则化参数 $\lambda$')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.xticks(sorted(list(set(reg_params) | {0, 0.25, 0.5, 0.75, 1}))) 
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ----------------------------------------------------------
# 图 4.2.1.2: KNN 的 K 值对鲁棒性的影响
# ----------------------------------------------------------
def plot_knn_k_comparison_line(data_list, dim, method='LDA', k_values=[1, 3, 5, 10], filename='figure_4_2_1_2.png'):    
    # 提取 K 值对应的 验证集 和 测试集 准确率
    val_accs = [get_data_point(data_list, dim, 'KNN', 'k', k, method, '验证准确率') for k in k_values]
    test_accs = [get_data_point(data_list, dim, 'KNN', 'k', k, method, '测试准确率') for k in k_values]
    
    plt.figure(figsize=(8, 5))
    
    # 绘制两条折线图
    plt.plot(k_values, val_accs, marker='o', linestyle='-', color='red', label='验证集准确率')
    plt.plot(k_values, test_accs, marker='x', linestyle='--', color='blue', label='测试集准确率')
    
    plt.ylim(0.8, 1.0)
    plt.title(f'图 4.2.1.2. KNN 近邻数 $K$ 对准确率的影响 ({method} D={dim})')
    plt.xlabel('近邻数 K')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--')
    plt.legend()
    
    # 添加标签 (只对测试集添加标签，避免图表过于拥挤)
    for k, acc in zip(k_values, test_accs):
        plt.text(k, acc + 0.005, f'{acc:.2%}', ha='center', va='bottom', fontsize=9, color='blue')
    
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ----------------------------------------------------------
# 图 4.2.2.1.1: - QDF(reg=0) vs KNN(K=10) 在 PCA 空间
# ----------------------------------------------------------
def plot_pca_dimensionality_qdf_knn_comparison(data_list, pca_dims, filename='figure_4_2_2_1_1.png'):
    
    # 定义要提取的参数
    REG_PARAM = 0
    K_PARAM = 10
    
    # 提取所有维度的 QDF 和 KNN 准确率
    qdf_val_accs = [get_data_point(data_list, d, 'QDF', 'reg', REG_PARAM, 'PCA', '验证准确率') for d in pca_dims]
    qdf_test_accs = [get_data_point(data_list, d, 'QDF', 'reg', REG_PARAM, 'PCA', '测试准确率') for d in pca_dims]
    knn_val_accs = [get_data_point(data_list, d, 'KNN', 'k', K_PARAM, 'PCA', '验证准确率') for d in pca_dims]
    knn_test_accs = [get_data_point(data_list, d, 'KNN', 'k', K_PARAM, 'PCA', '测试准确率') for d in pca_dims]

    plt.figure(figsize=(10, 6))

    # --- QDF 曲线 (颜色: 蓝色系) ---
    plt.plot(pca_dims, qdf_val_accs, marker='o', linestyle='-', color='blue', 
             label=f'QDF (reg={REG_PARAM}) 验证集')
    plt.plot(pca_dims, qdf_test_accs, marker='o', linestyle='--', color='skyblue', 
             label=f'QDF (reg={REG_PARAM}) 测试集')

    # --- KNN 曲线 (颜色: 红色系) ---
    plt.plot(pca_dims, knn_val_accs, marker='x', linestyle='-', color='red', 
             label=f'KNN (K={K_PARAM}) 验证集')
    plt.plot(pca_dims, knn_test_accs, marker='x', linestyle='--', color='lightcoral', 
             label=f'KNN (K={K_PARAM}) 测试集')
    
    # --- 样式设置 ---
    plt.title(f'图 4.2.2.1.1 性能对比: QDF(reg={REG_PARAM}) vs KNN(K={K_PARAM}) 随 $\\text{{PCA}}$ 维度的趋势')
    plt.xlabel('降维维度 (D)')
    plt.ylabel('准确率')
    plt.grid(True, linestyle='--')
    
    # 将所有准确率数据展平，用于确定 Y 轴范围
    all_accs = qdf_val_accs + qdf_test_accs + knn_val_accs + knn_test_accs
    if all_accs:
        y_min = max(0.0, min(all_accs) * 0.95)
        plt.ylim(y_min, 1.0)
    
    # 设置 X 轴刻度为所有维度
    plt.xticks(pca_dims)
    
    # 将图例置于底部，多列显示
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2) 
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------
# 图4.2.2.1.2: 维度趋势分析 - QDF(reg=0) vs KNN(K=10) 在 PCA 空间的时间对比
# ----------------------------------------------------------
def plot_pca_dimensionality_qdf_knn_time_comparison(data_list, pca_dims, filename='figure_4_2_2_1_2.png'):
    
    # 定义要提取的参数
    REG_PARAM = 0
    K_PARAM = 10
    
    # 提取所有维度的 QDF 和 KNN 测试时间
    qdf_times = [get_data_point(data_list, d, 'QDF', 'reg', REG_PARAM, 'PCA', '测试时间') for d in pca_dims]
    knn_times = [get_data_point(data_list, d, 'KNN', 'k', K_PARAM, 'PCA', '测试时间') for d in pca_dims]
    
    plt.figure(figsize=(10, 6))

    # --- QDF 曲线 (颜色: 蓝色实线) ---
    plt.plot(pca_dims, qdf_times, marker='o', linestyle='-', color='blue', 
             label=f'QDF (reg={REG_PARAM}) 测试时间')

    # --- KNN 曲线 (颜色: 红色实线) ---
    plt.plot(pca_dims, knn_times, marker='x', linestyle='-', color='red', 
             label=f'KNN (K={K_PARAM}) 测试时间')
    
    # --- 样式设置 ---
    plt.title(f'图 4.2.2.1.2 效率对比: QDF(reg={REG_PARAM}) vs KNN(K={K_PARAM}) 随 $\\text{{PCA}}$ 维度的测试时间趋势')
    plt.xlabel('降维维度 (D)')
    plt.ylabel('测试时间 (秒)')
    plt.grid(True, linestyle='--')
    
    # 添加标签
    for d, t in zip(pca_dims, qdf_times):
        plt.text(d, t + 0.05 * max(qdf_times), f'{t:.3f}s', ha='center', va='bottom', fontsize=8, color='blue')

    for d, t in zip(pca_dims, knn_times):
        plt.text(d, t + 0.05 * max(knn_times), f'{t:.3f}s', ha='center', va='bottom', fontsize=8, color='red')
        
    # 设置 X 轴刻度为所有维度
    plt.xticks(pca_dims)
    
    # 将图例置于底部，多列显示
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2) 
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# ==========================================================
# 3. 运行所有绘图
# ==========================================================
if __name__ == '__main__':
    print("开始生成实验结果图表...")
    
    # 选取 LDA D=9 作为代表子空间
    BEST_DIM = 9
    BEST_METHOD = 'LDA' 
    
    # 更新 KNN K 值范围
    KNN_K_VALUES = [1, 3, 5, 10]
    
    # 4.2.1. 同一子空间，同一分类器 
    if BEST_DIM in lda_dims:
        # 4.2.1.1 QDF 正则化分析
        plot_qdf_regularization(data, dim=BEST_DIM, method=BEST_METHOD)
        print(" -> 已生成图 4.2.1.1 (QDF $\lambda$ 稳定性)")
        
        # 4.2.1.2 KNN K 值对比 (使用折线图和新K值)
        plot_knn_k_comparison_line(data, dim=BEST_DIM, method=BEST_METHOD, k_values=KNN_K_VALUES)
        print(" -> 已生成图 4.2.1.2 (KNN K 值趋势分析)")
    else:
        print(f"警告：缺少 {BEST_METHOD} D={BEST_DIM} 数据，跳过 4.2.1 节分析绘图。")

    
    # --- 4.2.2.1.1 QDF vs KNN 随 PCA 维度的趋势分析 ---
    if len(pca_dims) >= 2:
        plot_pca_dimensionality_qdf_knn_comparison(
            data, 
            pca_dims=pca_dims, 
            filename='figure_4_2_2_1_1.png'
        )
        print(" -> 已生成图 4.2.2.1.1 (QDF vs KNN 随 PCA 维度的准确率趋势)")
    else:
        print("警告：PCA 维度点不足，跳过 4.2.3.0 绘图。")
        

    # --- 4.2.2.1.2 QDF vs KNN 随 PCA 维度的测试时间对比 ---
    if len(pca_dims) >= 2:
        plot_pca_dimensionality_qdf_knn_time_comparison(
            data, 
            pca_dims=pca_dims, 
            filename='figure_4_2_2_1_2.png'
        )
        print(" -> 已生成图 4.2.2.1.2 (QDF vs KNN 随 PCA 维度的测试时间对比)")
    else:
        print("警告：PCA 维度点不足，跳过 4.2.3.3 绘图。")

    print("所有图表生成完成！")