#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
消费者分群 - 深度聚类（Deep Embedded Clustering）
应用场景：基于购买动机（问题7）、口味偏好（问题8）、价格敏感度（问题15/30）进行细分
优势：比传统K-Means更好处理高维稀疏数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# 添加中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

def create_sample_data(n_samples=1000):
    """创建示例数据集"""
    # 购买动机选项
    motivation_options = ['口味', '价格', '包装', '品牌', '促销', '推荐', '广告', '习惯']
    # 口味偏好选项
    flavor_options = ['麻辣', '香辣', '酸辣', '咸香', '五香', '原味', '海鲜', '烧烤', '奶香', '水果']
    # 价格接受度范围
    price_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # 购买频率选项
    frequency_options = ['每天', '每周数次', '每周一次', '每月数次', '偶尔']
    
    # 创建空数据框
    df = pd.DataFrame()
    
    # 生成购买动机数据（多选）
    df['问题7'] = [np.random.choice(motivation_options, 
                                  size=np.random.randint(1, 4), 
                                  replace=False).tolist() for _ in range(n_samples)]
    
    # 生成口味偏好数据（多选）
    df['问题8'] = [np.random.choice(flavor_options, 
                                  size=np.random.randint(1, 5), 
                                  replace=False).tolist() for _ in range(n_samples)]
    
    # 生成价格接受度数据（单选，部分在问题15，部分在问题30）
    mask = np.random.choice([True, False], size=n_samples)
    df['问题15'] = np.where(mask, np.random.choice(price_range, size=n_samples), np.nan)
    df['问题30'] = np.where(~mask, np.random.choice(price_range, size=n_samples), np.nan)
    
    # 生成购买频率数据（单选）
    df['问题3'] = np.random.choice(frequency_options, size=n_samples)
    
    return df

def preprocess_data(df):
    """预处理数据"""
    print("预处理数据...")
    
    # 处理购买动机（多选题）
    motivation_columns = ['动机_便利性', '动机_品牌', '动机_价格', '动机_包装', '动机_口味', 
                         '动机_营养', '动机_新鲜度', '动机_促销活动']
    
    # 处理口味偏好（多选题）
    flavor_columns = ['口味_咸', '口味_甜', '口味_酸', '口味_辣', '口味_鲜', 
                     '口味_淡', '口味_重口味', '口味_清淡']
    
    # 创建多标签二值化器
    mlb_motivation = MultiLabelBinarizer(sparse_output=False)
    mlb_flavor = MultiLabelBinarizer(sparse_output=False)
    
    # 转换购买动机和口味偏好
    motivation_matrix = mlb_motivation.fit_transform(df['问题7'].apply(eval))
    flavor_matrix = mlb_flavor.fit_transform(df['问题8'].apply(eval))
    
    # 创建特征DataFrame
    motivation_df = pd.DataFrame(motivation_matrix, 
                               columns=[f'motivation_{col}' for col in mlb_motivation.classes_])
    flavor_df = pd.DataFrame(flavor_matrix,
                           columns=[f'flavor_{col}' for col in mlb_flavor.classes_])
    
    # 标准化价格承受度（使用MinMaxScaler而不是StandardScaler）
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    # 合并问题15和问题30的价格数据，使用问题15作为主要来源，问题30作为备选
    price_data = df['问题15'].combine_first(df['问题30']).values.reshape(-1, 1)
    price_tolerance = price_scaler.fit_transform(price_data)
    
    # 组合所有特征
    X = pd.concat([
        motivation_df * 2,  # 增加购买动机的权重
        flavor_df * 1.5,    # 增加口味偏好的权重
        pd.DataFrame({'price_tolerance': price_tolerance.flatten()})  # 保持价格承受度的原始权重
    ], axis=1)
    
    return X

def build_autoencoder(input_dim):
    """构建改进的自编码器"""
    # 编码器
    encoder = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(8, activation='relu')
    ])
    
    # 解码器
    decoder = Sequential([
        Dense(16, activation='relu', input_shape=(8,)),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(input_dim, activation='sigmoid')  # 使用sigmoid确保输出在0-1之间
    ])
    
    # 组合自编码器
    autoencoder = Sequential([encoder, decoder])
    
    # 使用Adam优化器，调整学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # 编译模型
    autoencoder.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return autoencoder, encoder

def train_autoencoder(autoencoder, X, epochs=150, batch_size=64, validation_split=0.2):
    """训练自编码器"""
    print("训练自编码器...")
    
    # 将DataFrame转换为numpy数组以避免索引问题
    X_array = X.values
    
    # 划分训练集和验证集
    X_train, X_val = train_test_split(X_array, test_size=validation_split, random_state=42)
    
    # 使用早停法防止过拟合
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # 训练自编码器
    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 可视化训练过程
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('自编码器训练过程')
    plt.xlabel('训练周期')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    
    # 绘制重构效果
    X_pred = autoencoder.predict(X_val)
    
    # 选择第一个特征进行可视化
    plt.subplot(1, 2, 2)
    plt.scatter(X_val[:, 0], X_pred[:, 0], alpha=0.5)
    plt.plot([-3, 3], [-3, 3], 'r--')
    plt.title('重构效果示例')
    plt.xlabel('原始值')
    plt.ylabel('重构值')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('autoencoder_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return history

def evaluate_clustering(X_scaled, max_clusters=10):
    """评估最佳聚类数"""
    print("评估最佳聚类数...")
    
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    inertias = []
    
    # 使用更多的初始化次数以获得更稳定的结果
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,  # 增加初始化次数
            max_iter=500  # 增加最大迭代次数
        )
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        calinski_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
        davies_scores.append(davies_bouldin_score(X_scaled, cluster_labels))
        inertias.append(kmeans.inertia_)
    
    # 计算综合得分
    silhouette_norm = MinMaxScaler().fit_transform(np.array(silhouette_scores).reshape(-1, 1)).flatten()
    calinski_norm = MinMaxScaler().fit_transform(np.array(calinski_scores).reshape(-1, 1)).flatten()
    davies_norm = 1 - MinMaxScaler().fit_transform(np.array(davies_scores).reshape(-1, 1)).flatten()
    
    # 使用加权平均计算最佳聚类数
    composite_scores = (silhouette_norm * 0.4 + calinski_norm * 0.3 + davies_norm * 0.3)
    best_n_clusters = np.argmax(composite_scores) + 2
    
    return best_n_clusters

def perform_clustering(latent_features, n_clusters, df):
    """执行聚类分析"""
    print(f"执行聚类分析，聚类数量: {n_clusters}...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_features)
    
    # 将聚类标签添加到原始数据框
    df['cluster'] = cluster_labels
    
    return df, kmeans

def visualize_clusters(latent_features, df):
    """使用t-SNE可视化聚类结果"""
    print("使用t-SNE可视化聚类结果...")
    
    # 使用t-SNE降维到2D空间进行可视化，调整参数以获得更好的分离效果
    print("正在执行t-SNE降维，这可能需要一些时间...")
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=30,  # 降低perplexity以减少计算量
        n_iter=1000,    # 减少迭代次数以加快计算
        learning_rate=200,  # 使用固定学习率
        init='pca',     # 使用PCA初始化以获得更好的起点
        verbose=1       # 添加进度输出
    )
    latent_2d = tsne.fit_transform(latent_features)
    print("t-SNE降维完成！")
    
    # 创建可视化数据框
    viz_df = pd.DataFrame({
        'x': latent_2d[:, 0],
        'y': latent_2d[:, 1],
        'cluster': df['cluster'].astype(str)
    })
    
    # 设置更好的颜色方案
    n_clusters = len(viz_df['cluster'].unique())
    colors = sns.color_palette('husl', n_clusters)
    
    # 创建更大的图形
    plt.figure(figsize=(12, 10))
    
    print("绘制聚类可视化图...")
    # 简化：移除核密度估计等高线，减少计算量
    # 直接绘制散点图
    sns.scatterplot(
        data=viz_df,
        x='x', y='y',
        hue='cluster',
        palette=colors,
        s=80,
        alpha=0.7,
        legend='full'
    )
    
    # 美化图表
    plt.title('消费者群体聚类分布 (t-SNE降维可视化)', fontsize=16, pad=20)
    plt.xlabel('t-SNE维度1', fontsize=14)
    plt.ylabel('t-SNE维度2', fontsize=14)
    
    # 调整图例
    plt.legend(
        title='消费者群体',
        title_fontsize=12,
        fontsize=10,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 调整布局以确保图例完全可见
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
    print("聚类可视化完成！")

def analyze_clusters(df, X):
    """分析各聚类群体特征"""
    print("分析各聚类群体特征...")
    
    # 将原始特征与聚类标签合并
    analysis_df = pd.concat([X, df[['cluster']]], axis=1)
    
    # 计算各聚类的特征均值
    cluster_means = analysis_df.groupby('cluster').mean()
    
    # 创建结果存储字典
    cluster_stats = {}
    
    # 分析每个聚类的特征
    for cluster in sorted(df['cluster'].unique()):
        # 获取当前聚类的数据
        cluster_analysis_data = analysis_df[analysis_df['cluster'] == cluster]
        
        # 计算样本数量和占比
        sample_count = len(cluster_analysis_data)
        sample_ratio = sample_count / len(df) * 100
        
        # 计算价格接受度的均值和标准差（从预处理后的特征矩阵X中获取）
        price_mean = cluster_analysis_data['price_tolerance'].mean()
        price_std = cluster_analysis_data['price_tolerance'].std()
        
        # 提取购买动机
        motivation_cols = [col for col in X.columns if col.startswith('motivation_')]
        motivation_means = cluster_means.loc[cluster, motivation_cols]
        top_motivations = motivation_means.nlargest(3)
        
        # 提取偏好口味
        flavor_cols = [col for col in X.columns if col.startswith('flavor_')]
        flavor_means = cluster_means.loc[cluster, flavor_cols]
        top_flavors = flavor_means.nlargest(3)
        
        print(f"\n===== 群体{cluster}分析 =====")
        print(f"样本数量: {sample_count} ({sample_ratio:.1f}%)")
        print(f"价格承受度: {price_mean:.2f} ± {price_std:.2f}")
        
        # 打印主要购买动机
        print("\n主要购买动机:")
        for motivation, score in top_motivations.items():
            motivation_name = motivation.replace('motivation_', '')
            print(f"- {motivation_name}: {score:.2f}")
        
        # 打印偏好口味
        print("\n偏好口味:")
        for flavor, score in top_flavors.items():
            flavor_name = flavor.replace('flavor_', '')
            print(f"- {flavor_name}: {score:.2f}")
        
        # 存储结果
        cluster_stats[cluster] = {
            'sample_count': sample_count,
            'sample_ratio': sample_ratio,
            'price_mean': price_mean,
            'price_std': price_std,
            'top_motivations': [(m.replace('motivation_', ''), s) for m, s in top_motivations.items()],
            'top_flavors': [(f.replace('flavor_', ''), s) for f, s in top_flavors.items()]
        }
    
    # 可视化各聚类的价格接受度分布
    print("\n绘制价格承受度分布图...")
    plt.figure(figsize=(12, 8))
    
    # 创建箱线图
    box_plot = sns.boxplot(
        x='cluster',
        y='price_tolerance',
        data=analysis_df,  # 使用包含price_tolerance的analysis_df
        palette='husl',
        width=0.7,
        fliersize=5,
        linewidth=1.5
    )
    
    # 添加群体均值点
    sns.pointplot(
        x='cluster',
        y='price_tolerance',
        data=analysis_df,  # 使用包含price_tolerance的analysis_df
        color='red',
        markers='D',
        scale=0.7,
        ci=None
    )
    
    # 添加数值标注
    for i, cluster in enumerate(sorted(analysis_df['cluster'].unique())):
        cluster_data = analysis_df[analysis_df['cluster'] == cluster]['price_tolerance']
        mean_val = cluster_data.mean()
        median_val = cluster_data.median()
        
        # 在箱线图上方添加均值和中位数标注
        plt.text(
            i, analysis_df['price_tolerance'].max() + 0.05,
            f'均值: {mean_val:.2f}\n中位数: {median_val:.2f}',
            ha='center',
            va='bottom',
            fontsize=9
        )
    
    # 美化图表
    plt.title('各消费者群体价格承受度分布', fontsize=16, pad=20)
    plt.xlabel('消费者群体', fontsize=14)
    plt.ylabel('价格承受度', fontsize=14)
    
    # 设置坐标轴范围，确保标注可见
    plt.ylim(0, analysis_df['price_tolerance'].max() * 1.15)
    
    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)
    
    # 添加图例
    plt.legend(['群体均值'], loc='upper right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig('price_tolerance_by_cluster.png', dpi=300, bbox_inches='tight')
    print("价格承受度分布图已保存")
    
    # 可视化各聚类的主要特征（热力图）
    print("\n绘制特征热力图...")
    # 标准化聚类特征均值以便更好地可视化差异
    scaler_viz = MinMaxScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler_viz.fit_transform(cluster_means),
        index=cluster_means.index,
        columns=cluster_means.columns
    )
    
    # 重命名特征列，使其更简洁易读
    feature_rename = {
        'motivation_': '动机_',
        'flavor_': '口味_',
        'price_tolerance': '价格接受度'
    }
    
    cluster_means_scaled.columns = [
        next((new + col[len(old):] for old, new in feature_rename.items() if col.startswith(old)), col)
        for col in cluster_means_scaled.columns
    ]
    
    plt.figure(figsize=(20, 10))
    ax = sns.heatmap(
        cluster_means_scaled,
        cmap='RdYlBu_r',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': '标准化特征值'},
        square=True
    )
    
    plt.title('消费者群体特征分布热力图', fontsize=16, pad=20)
    plt.ylabel('聚类编号', fontsize=14)
    plt.xlabel('特征维度', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
    print("特征热力图已保存")
    
    return cluster_stats

def interpret_clusters(cluster_profile, n_clusters):
    """解释聚类结果"""
    print("\n===== 聚类群体解读 =====")
    
    # 这里是示例解读，实际应用中需要根据具体数据进行解读
    cluster_interpretations = {
        0: "价格敏感型：价格接受度较低，注重促销活动，偏好经典口味。营销策略：推荐促销组合。",
        1: "品质追求型：高价格接受度，关注健康成分。营销策略：推送健康版新品信息。",
        2: "猎奇尝新型：偏好新奇特口味，联名款接受度高。营销策略：优先推送联名产品。",
        3: "怀旧忠诚型：复购率高，偏好传统包装。营销策略：强化情怀营销。",
        4: "便利实用型：注重便捷性，价格适中。营销策略：强调便携和易用性。"
    }
    
    # 根据实际聚类数量输出解读
    for i in range(n_clusters):
        if i in cluster_interpretations:
            print(f"群组{i}：{cluster_interpretations[i]}")
        else:
            print(f"群组{i}：需要根据数据特征进行具体解读")

def main():
    """主函数"""
    print("=== 消费者分群分析 - 深度聚类 ===\n")
    
    # 加载数据
    try:
        df = pd.read_csv('consumer_data.csv')
        print("成功加载数据")
    except FileNotFoundError:
        print("创建示例数据...")
        df = create_sample_data(n_samples=1000)
        df.to_csv('consumer_data.csv', index=False)
    
    # 数据预处理
    X = preprocess_data(df)
    
    # 构建并训练自编码器
    autoencoder, encoder = build_autoencoder(X.shape[1])
    history = train_autoencoder(autoencoder, X)
    
    # 提取潜在特征
    latent_features = encoder.predict(X)
    
    # 评估最佳聚类数
    optimal_k = evaluate_clustering(X)
    
    # 执行聚类
    df, kmeans = perform_clustering(latent_features, optimal_k, df)
    
    # 可视化与分析
    visualize_clusters(latent_features, df)
    cluster_stats = analyze_clusters(df, X)
    
    # 输出分析结果
    print("\n=== 聚类分析结果 ===")
    for group, stats in cluster_stats.items():
        print(f"\n{group}:")
        print(f"样本数量: {stats['sample_count']} ({stats['sample_ratio']:.1f}%)")
        print(f"价格承受度: {stats['price_mean']:.2f} ± {stats['price_std']:.2f}")
        print(f"主要购买动机: {', '.join(x[0] for x in stats['top_motivations'])}")
        print(f"偏好口味: {', '.join(x[0] for x in stats['top_flavors'])}")
    
    print("\n分析完成！所有可视化结果已保存。")

if __name__ == "__main__":
    main()
