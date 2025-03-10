#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
渠道关联分析 - 网络图模型
应用场景：分析认知渠道（问题5/23）与购买渠道（问题19/33）的关联
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib
import os
import seaborn as sns
from pathlib import Path

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def load_data(file_path):
    """加载数据"""
    try:
        # 尝试加载Excel文件
        df = pd.read_excel(file_path)
        print(f"成功加载数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"加载数据失败: {e}")
        # 如果无法加载，创建示例数据
        return create_sample_data()

def create_sample_data(n_samples=500):
    """创建示例数据"""
    print("创建示例数据...")
    
    # 认知渠道选项
    awareness_channels = [
        '朋友推荐', '社交媒体', '电商平台推荐', '线下广告', 
        '短视频平台', '直播带货', '搜索引擎', '其他'
    ]
    
    # 购买渠道选项
    purchase_channels = [
        '淘宝/天猫', '京东', '拼多多', '线下超市', 
        '便利店', '直播平台', '微信小程序', '其他'
    ]
    
    # 生成随机数据
    data = {
        '认知渠道': np.random.choice(awareness_channels, n_samples),
        '购买渠道': np.random.choice(purchase_channels, n_samples)
    }
    
    # 创建一些关联性更强的数据
    # 例如：通过社交媒体了解的更可能在淘宝购买
    for i in range(100):
        data['认知渠道'][i] = '社交媒体'
        data['购买渠道'][i] = '淘宝/天猫'
    
    # 通过直播了解的更可能在直播平台购买
    for i in range(100, 200):
        data['认知渠道'][i] = '直播带货'
        data['购买渠道'][i] = '直播平台'
    
    # 通过朋友推荐的更可能在线下购买
    for i in range(200, 300):
        data['认知渠道'][i] = '朋友推荐'
        data['购买渠道'][i] = '线下超市'
    
    df = pd.DataFrame(data)
    print(f"示例数据创建完成，共 {len(df)} 条记录")
    return df

def preprocess_data(df):
    """数据预处理"""
    # 检查列名，尝试找到认知渠道和购买渠道的列
    awareness_cols = [col for col in df.columns if '认知' in col or '了解' in col]
    purchase_cols = [col for col in df.columns if '购买' in col or '购物' in col]
    
    # 如果找不到，尝试使用问题编号
    if not awareness_cols:
        awareness_cols = [col for col in df.columns if '问题5' in col or '问题23' in col]
    if not purchase_cols:
        purchase_cols = [col for col in df.columns if '问题19' in col or '问题33' in col]
    
    # 如果仍然找不到，使用默认列名
    if not awareness_cols:
        awareness_cols = ['认知渠道']
    if not purchase_cols:
        purchase_cols = ['购买渠道']
    
    # 使用找到的第一个列
    awareness_col = awareness_cols[0]
    purchase_col = purchase_cols[0]
    
    print(f"使用列: 认知渠道='{awareness_col}', 购买渠道='{purchase_col}'")
    
    # 重命名列以便后续处理
    df = df.rename(columns={awareness_col: '认知渠道', purchase_col: '购买渠道'})
    
    # 处理多选问题（如果是字符串形式的多选）
    if df['认知渠道'].dtype == 'object':
        df['认知渠道'] = df['认知渠道'].str.split('[,;，；]').str[0]
    if df['购买渠道'].dtype == 'object':
        df['购买渠道'] = df['购买渠道'].str.split('[,;，；]').str[0]
    
    # 删除缺失值
    df = df.dropna(subset=['认知渠道', '购买渠道'])
    
    return df

def create_network_graph(df, output_dir):
    """创建网络图"""
    # 构建共现矩阵
    channel_matrix = pd.crosstab(df['认知渠道'], df['购买渠道'])
    print("渠道共现矩阵:")
    print(channel_matrix)
    
    # 创建有向网络图
    G = nx.DiGraph()
    
    # 添加节点
    awareness_channels = df['认知渠道'].unique()
    purchase_channels = df['购买渠道'].unique()
    
    # 添加认知渠道节点
    for channel in awareness_channels:
        G.add_node(f"认知: {channel}", type='awareness', color='skyblue')
    
    # 添加购买渠道节点
    for channel in purchase_channels:
        G.add_node(f"购买: {channel}", type='purchase', color='lightgreen')
    
    # 添加边
    for i, awareness in enumerate(channel_matrix.index):
        for j, purchase in enumerate(channel_matrix.columns):
            weight = channel_matrix.iloc[i, j]
            if weight > 0:
                G.add_edge(
                    f"认知: {awareness}", 
                    f"购买: {purchase}", 
                    weight=weight
                )
    
    # 绘制网络图
    plt.figure(figsize=(14, 10))
    
    # 设置节点位置
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # 获取节点颜色
    node_colors = [G.nodes[node].get('color', 'gray') for node in G.nodes()]
    
    # 获取边的权重
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    normalized_weights = [3 * w / max_weight for w in edge_weights]
    
    # 绘制节点
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=2000, 
        node_color=node_colors,
        alpha=0.8
    )
    
    # 绘制边
    nx.draw_networkx_edges(
        G, pos, 
        width=normalized_weights,
        edge_color='gray',
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle='->'
    )
    
    # 绘制标签
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_family='SimHei',
        font_weight='bold'
    )
    
    plt.title('认知渠道与购买渠道关联网络图', fontsize=16, pad=20)
    plt.axis('off')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_network_basic.png'), dpi=300, bbox_inches='tight')
    
    # 创建更高级的可视化
    plt.figure(figsize=(16, 12))
    
    # 使用圆形布局
    pos = nx.circular_layout(G)
    
    # 计算节点大小（基于连接数）
    node_sizes = [3000 * (1 + G.degree(node) / 10) for node in G.nodes()]
    
    # 绘制节点
    awareness_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'awareness']
    purchase_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'purchase']
    
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=awareness_nodes,
        node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in awareness_nodes],
        node_color='skyblue',
        alpha=0.8,
        label='认知渠道'
    )
    
    nx.draw_networkx_nodes(
        G, pos, 
        nodelist=purchase_nodes,
        node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in purchase_nodes],
        node_color='lightgreen',
        alpha=0.8,
        label='购买渠道'
    )
    
    # 绘制边（宽度基于权重）
    edge_widths = [5 * G[u][v]['weight'] / max_weight for u, v in G.edges()]
    
    nx.draw_networkx_edges(
        G, pos, 
        width=edge_widths,
        edge_color='gray',
        alpha=0.6,
        arrows=True,
        arrowsize=20,
        arrowstyle='->'
    )
    
    # 绘制标签
    nx.draw_networkx_labels(
        G, pos, 
        font_size=11, 
        font_family='SimHei',
        font_weight='bold'
    )
    
    plt.title('认知渠道与购买渠道关联网络图 (圆形布局)', fontsize=18, pad=20)
    plt.axis('off')
    
    # 修改图例位置和样式，解决重叠问题
    legend = plt.legend(
        fontsize=12,
        loc='upper right',  # 位置保持在右上角
        bbox_to_anchor=(1.15, 1.0),  # 向右移动图例
        frameon=True,  # 添加边框
        facecolor='white',  # 白色背景
        edgecolor='gray',  # 灰色边框
        title='节点类型',  # 添加图例标题
        title_fontsize=14
    )
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_network_circular.png'), dpi=300, bbox_inches='tight')
    
    # 创建热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        channel_matrix, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='d',
        linewidths=0.5
    )
    plt.title('认知渠道与购买渠道关联热力图', fontsize=16)
    plt.xlabel('购买渠道', fontsize=12)
    plt.ylabel('认知渠道', fontsize=12)
    
    # 保存热力图
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_heatmap.png'), dpi=300, bbox_inches='tight')
    
    return G, channel_matrix

def analyze_channel_relationships(G, channel_matrix, output_dir):
    """分析渠道关系"""
    # 计算每个认知渠道的转化率
    conversion_rates = {}
    for awareness in channel_matrix.index:
        total = channel_matrix.loc[awareness].sum()
        rates = {}
        for purchase in channel_matrix.columns:
            count = channel_matrix.loc[awareness, purchase]
            rate = count / total if total > 0 else 0
            rates[purchase] = rate
        conversion_rates[awareness] = rates
    
    # 创建转化率数据框
    conversion_df = pd.DataFrame(conversion_rates).T
    
    # 绘制转化率热力图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conversion_df, 
        annot=True, 
        cmap='YlOrRd', 
        fmt='.2f',
        linewidths=0.5,
        vmin=0, 
        vmax=1
    )
    plt.title('认知渠道到购买渠道的转化率热力图', fontsize=16)
    plt.xlabel('购买渠道', fontsize=12)
    plt.ylabel('认知渠道', fontsize=12)
    
    # 保存转化率热力图
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'channel_conversion_heatmap.png'), dpi=300, bbox_inches='tight')
    
    # 计算每个渠道的中心性指标
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G, weight='weight')  # 考虑边的权重
    
    # 尝试计算特征向量中心性，如果失败则使用PageRank作为替代
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000, tol=1.0e-6)
        has_eigenvector = True
    except nx.PowerIterationFailedConvergence:
        print("特征向量中心性计算未收敛，使用PageRank作为替代")
        eigenvector_centrality = nx.pagerank(G, weight='weight')
        has_eigenvector = True
    except Exception as e:
        print(f"计算特征向量中心性时出错: {e}")
        has_eigenvector = False
    
    # 创建中心性指标数据框
    if has_eigenvector:
        centrality_df = pd.DataFrame({
            '节点': list(degree_centrality.keys()),
            '度中心性': list(degree_centrality.values()),
            '介数中心性': list(betweenness_centrality.values()),
            'PageRank中心性': list(eigenvector_centrality.values())
        })
    else:
        centrality_df = pd.DataFrame({
            '节点': list(degree_centrality.keys()),
            '度中心性': list(degree_centrality.values()),
            '介数中心性': list(betweenness_centrality.values())
        })
    
    # 归一化处理
    for col in centrality_df.columns:
        if col != '节点' and centrality_df[col].max() > 0:  # 避免除以零
            centrality_df[col] = centrality_df[col] / centrality_df[col].max()
    
    # 按度中心性排序
    centrality_df = centrality_df.sort_values('度中心性', ascending=False)
    
    # 绘制中心性指标条形图
    plt.figure(figsize=(15, 8))
    
    # 度中心性
    plt.subplot(1, 2, 1)
    top_10_degree = centrality_df.head(10)
    sns.barplot(
        x='度中心性', 
        y='节点', 
        data=top_10_degree,
        palette='viridis'
    )
    plt.title('渠道度中心性 Top 10', fontsize=14)
    plt.xlabel('归一化度中心性', fontsize=12)
    plt.ylabel('渠道', fontsize=12)
    
    # 介数中心性
    plt.subplot(1, 2, 2)
    top_10_betweenness = centrality_df.nlargest(10, '介数中心性')
    sns.barplot(
        x='介数中心性', 
        y='节点', 
        data=top_10_betweenness,
        palette='viridis'
    )
    plt.title('渠道介数中心性 Top 10', fontsize=14)
    plt.xlabel('归一化介数中心性', fontsize=12)
    plt.ylabel('渠道', fontsize=12)
    
    # 调整布局避免标签重叠
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(output_dir, 'channel_centrality.png'), dpi=300, bbox_inches='tight')
    
    return conversion_df, centrality_df

def main():
    """主函数"""
    print("=== 渠道关联分析 - 网络图模型 ===")
    
    # 创建输出目录
    output_dir = "渠道关联分析"
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试加载数据
    try:
        # 首先尝试从默认路径加载
        file_path = "C:\\Users\\jiawang\\Desktop\\狗牙儿.xlsx"
        if not os.path.exists(file_path):
            # 如果默认路径不存在，尝试在当前目录查找Excel文件
            excel_files = list(Path('.').glob('*.xlsx'))
            if excel_files:
                file_path = str(excel_files[0])
            else:
                # 如果找不到Excel文件，使用示例数据
                df = create_sample_data()
        else:
            df = load_data(file_path)
    except Exception as e:
        print(f"加载数据出错: {e}")
        df = create_sample_data()
    
    # 数据预处理
    df = preprocess_data(df)
    
    # 创建网络图
    G, channel_matrix = create_network_graph(df, output_dir)
    
    # 分析渠道关系
    conversion_df, centrality_df = analyze_channel_relationships(G, channel_matrix, output_dir)
    
    # 输出分析结果
    print("\n=== 分析结果 ===")
    print("\n认知渠道到购买渠道的转化率:")
    print(conversion_df)
    
    print("\n渠道中心性指标:")
    print(centrality_df.head(10))
    
    print(f"\n分析完成！所有图表已保存到 '{output_dir}' 目录")

if __name__ == "__main__":
    # 执行主函数
    main() 