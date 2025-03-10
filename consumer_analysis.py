"""
狗牙儿零食品牌消费者画像分析
使用高级数据分析和可视化方法分析消费者特征、消费频率和品牌认知度

作者：[Your Name]
日期：[Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from wordcloud import WordCloud
import jieba
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

# 设置Plotly输出为浏览器
pio.renderers.default = "browser"

# 自定义颜色方案
colors = px.colors.qualitative.Bold
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#FF9999", "#FF3333", "#990000"])

def show_excel_headers(file_path):
    """
    读取并显示Excel文件的标题行（第一行）
    
    参数:
        file_path (str): Excel文件路径
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件 '{file_path}' 不存在")
            return None
        
        # 读取Excel文件
        print(f"正在读取文件: {file_path}")
        df = pd.read_excel(file_path)
        
        # 获取并显示标题行
        headers = df.columns.tolist()
        
        print("\n文件标题行（第一行）:")
        print("-" * 100)
        for i, header in enumerate(headers):
            print(f"{i+1}. {header}")
        print("-" * 100)
        
        # 显示数据形状
        print(f"\n数据形状: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        return df
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return None

def load_data(file_path):
    """
    加载Excel数据并进行预处理
    
    参数:
        file_path (str): Excel文件路径
        
    返回:
        pd.DataFrame: 处理后的数据框
    """
    print(f"正在加载数据: {file_path}")
    df = pd.read_excel(file_path)
    
    # 显示所有列名
    print("\n原始数据列名:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    # 找到性别、年龄、购买频率和品牌认知度的列索引
    gender_col = None
    age_col = None
    freq_col = None
    awareness_col = None
    
    # 根据列名关键词查找相关列
    for i, col in enumerate(df.columns):
        col_lower = str(col).lower()
        if '性别' in col_lower:
            gender_col = i
        elif '年龄' in col_lower:
            age_col = i
        elif '购买' in col_lower and '频率' in col_lower:
            freq_col = i
        elif ('听说' in col_lower or '认知' in col_lower) and '狗牙儿' in col_lower:
            awareness_col = i
    
    # 如果找不到列，使用默认的前四列
    if gender_col is None:
        gender_col = 0
        print("未找到性别列，使用第1列作为性别")
    else:
        print(f"使用第{gender_col+1}列 '{df.columns[gender_col]}' 作为性别")
        
    if age_col is None:
        age_col = 1
        print("未找到年龄列，使用第2列作为年龄")
    else:
        print(f"使用第{age_col+1}列 '{df.columns[age_col]}' 作为年龄")
        
    if freq_col is None:
        freq_col = 2
        print("未找到购买频率列，使用第3列作为购买频率")
    else:
        print(f"使用第{freq_col+1}列 '{df.columns[freq_col]}' 作为购买频率")
        
    if awareness_col is None:
        awareness_col = 3
        print("未找到品牌认知度列，使用第4列作为品牌认知度")
    else:
        print(f"使用第{awareness_col+1}列 '{df.columns[awareness_col]}' 作为品牌认知度")
    
    # 创建一个新的DataFrame，只包含我们需要的列
    analysis_df = pd.DataFrame({
        "性别": df.iloc[:, gender_col],
        "年龄": df.iloc[:, age_col],
        "购买频率": df.iloc[:, freq_col],
        "品牌认知度": df.iloc[:, awareness_col]
    })
    
    # 数据清洗
    # 将性别编码为数值
    # 检查性别的唯一值
    print("\n性别唯一值:")
    print(analysis_df["性别"].unique())
    
    # 根据实际数据调整性别映射
    gender_mapping = {
        "男": 0, "女": 1, 
        1: 0, 2: 1,  # 数值编码
        "男性": 0, "女性": 1,  # 可能的其他表示
        "1": 0, "2": 1  # 字符串形式的数值
    }
    
    # 尝试映射性别
    analysis_df["性别_数值"] = analysis_df["性别"].astype(str).map(gender_mapping)
    # 如果映射失败，使用默认值
    if analysis_df["性别_数值"].isna().any():
        print("警告: 部分性别值无法映射，将使用默认值0")
        analysis_df["性别_数值"] = analysis_df["性别_数值"].fillna(0)
    
    # 处理年龄 - 确保是数值
    # 检查年龄的唯一值
    print("\n年龄唯一值:")
    print(analysis_df["年龄"].unique()[:10])  # 只显示前10个，避免太多
    
    try:
        analysis_df["年龄"] = pd.to_numeric(analysis_df["年龄"], errors='coerce')
        # 填充缺失值
        median_age = analysis_df["年龄"].median()
        if pd.isna(median_age):  # 如果中位数也是NaN
            median_age = 25  # 使用默认值
        analysis_df["年龄"].fillna(median_age, inplace=True)
        print(f"年龄中位数: {median_age}")
    except:
        print("警告: 年龄列转换为数值时出现问题，将使用默认值")
        analysis_df["年龄"] = 25  # 使用默认值
    
    # 将年龄分组
    age_bins = [0, 18, 25, 35, 45, 100]
    age_labels = ["18岁以下", "18-25岁", "26-35岁", "36-45岁", "45岁以上"]
    analysis_df["年龄段"] = pd.cut(analysis_df["年龄"], bins=age_bins, labels=age_labels)
    
    # 处理购买频率
    # 检查购买频率的唯一值
    print("\n购买频率唯一值:")
    print(analysis_df["购买频率"].unique())
    
    # 根据实际数据创建购买频率映射
    freq_mapping = {
        "从不": 0,
        "很少（每月少于1次）": 1,
        "偶尔（每月1-2次）": 2,
        "经常（每周1-2次）": 3,
        "频繁（每周3次以上）": 4,
        " 偶尔购买": 1,  # 根据实际数据调整
        " 每月1-2次": 2,
        " 每周2-3次": 3,
        " 每天": 4,
        "偶尔购买": 1,
        "每月1-2次": 2,
        "每周2-3次": 3,
        "每天": 4,
        "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,  # 字符串形式的数值
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4  # 数值编码
    }
    
    # 尝试映射购买频率
    analysis_df["购买频率_数值"] = analysis_df["购买频率"].astype(str).map(freq_mapping)
    # 如果映射失败，使用默认值
    if analysis_df["购买频率_数值"].isna().any():
        print("警告: 部分购买频率值无法映射，将使用默认值2")
        analysis_df["购买频率_数值"] = analysis_df["购买频率_数值"].fillna(2)
    
    # 处理品牌认知度
    # 检查品牌认知度的唯一值
    print("\n品牌认知度唯一值:")
    print(analysis_df["品牌认知度"].unique())
    
    # 根据实际数据创建品牌认知度映射
    awareness_mapping = {
        "从未听说过": 0,
        "听说过但未购买": 1,
        "购买过1-2次": 2,
        "经常购买": 3,
        "忠实粉丝": 4,
        "否": 0,  # 根据实际数据调整
        "是": 2,  # 假设"是"表示知道并可能购买过
        "不知道": 0,
        "知道但未购买": 1,
        "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,  # 字符串形式的数值
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4  # 数值编码
    }
    
    # 尝试映射品牌认知度
    analysis_df["品牌认知度_数值"] = analysis_df["品牌认知度"].astype(str).map(awareness_mapping)
    # 如果映射失败，使用默认值
    if analysis_df["品牌认知度_数值"].isna().any():
        print("警告: 部分品牌认知度值无法映射，将使用默认值1")
        analysis_df["品牌认知度_数值"] = analysis_df["品牌认知度_数值"].fillna(1)
    
    # 确保所有必要的列都有值
    analysis_df = analysis_df.fillna({
        "性别_数值": 0,
        "年龄": 25,
        "购买频率_数值": 2,
        "品牌认知度_数值": 1
    })
    
    print("数据加载完成，共 {} 行记录".format(len(analysis_df)))
    return analysis_df

def analyze_consumer_profile(df):
    """
    分析消费者画像
    
    参数:
        df (pd.DataFrame): 数据框
    """
    print("\n===== 消费者画像分析 =====")
    
    # 1. 性别分布分析
    plt.figure(figsize=(18, 12))
    
    plt.subplot(2, 3, 1)
    gender_counts = df["性别_数值"].map({0: "男", 1: "女"}).value_counts()
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', 
            colors=sns.color_palette("Set3"), startangle=90)
    plt.title("受访者性别分布", fontsize=14)
    
    # 2. 年龄分布分析
    plt.subplot(2, 3, 2)
    age_counts = df["年龄段"].value_counts().sort_index()
    sns.barplot(x=age_counts.index, y=age_counts.values, palette="viridis")
    plt.title("受访者年龄分布", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("人数")
    
    # 3. 购买频率分析
    plt.subplot(2, 3, 3)
    freq_counts = df["购买频率_数值"].value_counts().sort_index()
    freq_labels = ["从不", "很少", "偶尔", "经常", "频繁"]
    freq_labels = freq_labels[:len(freq_counts)]  # 确保标签数量匹配
    sns.barplot(x=freq_labels, y=freq_counts.values, palette="plasma")
    plt.title("休闲零食购买频率分布", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("人数")
    
    # 4. 品牌认知度分析
    plt.subplot(2, 3, 4)
    awareness_counts = df["品牌认知度_数值"].value_counts().sort_index()
    awareness_labels = ["不知道", "听说过", "购买过", "经常购买", "忠实粉丝"]
    awareness_labels = awareness_labels[:len(awareness_counts)]  # 确保标签数量匹配
    sns.barplot(x=awareness_labels, y=awareness_counts.values, palette="magma")
    plt.title("狗牙儿品牌认知度分布", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("人数")
    
    # 5. 年龄与购买频率的关系
    plt.subplot(2, 3, 5)
    age_freq = df.groupby("年龄段")["购买频率_数值"].mean().reset_index()
    sns.barplot(x="年龄段", y="购买频率_数值", data=age_freq, palette="crest")
    plt.title("不同年龄段的购买频率", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("平均购买频率")
    
    # 6. 年龄与品牌认知度的关系
    plt.subplot(2, 3, 6)
    age_awareness = df.groupby("年龄段")["品牌认知度_数值"].mean().reset_index()
    sns.barplot(x="年龄段", y="品牌认知度_数值", data=age_awareness, palette="mako")
    plt.title("不同年龄段的品牌认知度", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("平均品牌认知度")
    
    plt.tight_layout()
    plt.savefig("消费者画像基础分析.png", dpi=300, bbox_inches="tight")
    print("已保存消费者画像基础分析图表")

def create_advanced_visualizations(df):
    """
    创建高级可视化图表
    
    参数:
        df (pd.DataFrame): 数据框
    """
    print("\n===== 创建高级可视化 =====")
    
    # 1. 使用Plotly创建交互式热图
    # 计算相关矩阵
    corr_columns = ["性别_数值", "年龄", "购买频率_数值", "品牌认知度_数值"]
    corr_matrix = df[corr_columns].corr()
    
    # 重命名列以便更好地显示
    corr_matrix.index = ["性别", "年龄", "购买频率", "品牌认知度"]
    corr_matrix.columns = ["性别", "年龄", "购买频率", "品牌认知度"]
    
    # 创建热图 - 使用Matplotlib (更清晰的静态图)
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角掩码
    cmap = sns.diverging_palette(230, 20, as_cmap=True)  # 创建发散色调
    
    # 绘制热图
    sns.heatmap(corr_matrix, 
                annot=True,  # 显示数值
                fmt=".2f",   # 数值格式
                cmap=cmap,   # 颜色映射
                square=True, # 正方形单元格
                linewidths=.5, # 网格线宽度
                cbar_kws={"shrink": .8, "label": "相关系数"}, # 颜色条设置
                vmin=-1, vmax=1) # 值范围
    
    plt.title("消费者特征相关性热图", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("消费者特征相关性热图.png", dpi=300, bbox_inches="tight")
    print("已保存消费者特征相关性热图(静态版)")
    
    # 创建交互式热图 - 使用Plotly
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    color_continuous_scale='RdBu_r',
                    labels=dict(x="特征", y="特征", color="相关系数"),
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    title="消费者特征相关性热图")
    
    fig.update_layout(
        width=800,
        height=800,
        title_font_size=20,
        font_size=14
    )
    
    fig.write_html("消费者特征相关性热图.html")
    print("已保存消费者特征相关性热图(交互版)")
    
    # 2. 创建3D散点图 - 年龄、购买频率和品牌认知度的关系
    fig = px.scatter_3d(df, 
                        x="年龄", 
                        y="购买频率_数值", 
                        z="品牌认知度_数值",
                        color="性别_数值", 
                        color_discrete_map={0: "blue", 1: "red"},
                        hover_name="年龄段",
                        opacity=0.7,
                        labels={
                            "年龄": "年龄",
                            "购买频率_数值": "购买频率",
                            "品牌认知度_数值": "品牌认知度",
                            "性别_数值": "性别"
                        },
                        title="年龄、购买频率与品牌认知度的3D关系")
    
    fig.update_layout(
        scene=dict(
            xaxis_title="年龄",
            yaxis_title="购买频率",
            zaxis_title="品牌认知度"),
        legend=dict(
            title="性别", 
            itemsizing="constant",
            itemtext=["男性", "女性"]
        ),
        width=900,
        height=700
    )
    
    # 更新颜色映射的标签
    fig.update_traces(
        selector=dict(name="0"),
        name="男性"
    )
    fig.update_traces(
        selector=dict(name="1"),
        name="女性"
    )
    
    fig.write_html("消费者特征3D关系图.html")
    print("已保存消费者特征3D关系图")
    
    # 3. 创建雷达图 - 不同年龄段的消费特征
    # 计算不同年龄段的平均值
    radar_data = df.groupby("年龄段")[["购买频率_数值", "品牌认知度_数值"]].mean()
    
    # 标准化数据
    scaler = StandardScaler()
    radar_data_scaled = pd.DataFrame(scaler.fit_transform(radar_data), 
                                    index=radar_data.index, 
                                    columns=radar_data.columns)
    
    # 创建雷达图
    categories = ["购买频率", "品牌认知度"]
    fig = go.Figure()
    
    # 自定义颜色
    colors = px.colors.qualitative.Plotly
    
    for i, age_group in enumerate(radar_data_scaled.index):
        fig.add_trace(go.Scatterpolar(
            r=[radar_data_scaled.loc[age_group, "购买频率_数值"], 
               radar_data_scaled.loc[age_group, "品牌认知度_数值"],
               radar_data_scaled.loc[age_group, "购买频率_数值"]],  # 闭合雷达图
            theta=categories + [categories[0]],  # 闭合雷达图
            fill='toself',
            name=age_group,
            line=dict(color=colors[i % len(colors)]),
            opacity=0.8
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-2, 2],  # 标准化后的范围
                tickfont=dict(size=12),
                title="标准化得分",
                titlefont=dict(size=14)
            ),
            angularaxis=dict(
                tickfont=dict(size=14)
            )
        ),
        title=dict(
            text="不同年龄段的消费特征雷达图",
            font=dict(size=20)
        ),
        legend=dict(
            title="年龄段",
            font=dict(size=12)
        ),
        showlegend=True,
        width=800,
        height=600
    )
    
    fig.write_html("不同年龄段消费特征雷达图.html")
    print("已保存不同年龄段消费特征雷达图")
    
    # 4. 创建桑基图 - 性别、年龄段与购买频率的流向
    # 准备桑基图数据
    sankey_df = df.copy()
    sankey_df["性别_数值"] = sankey_df["性别_数值"].map({0: "男性", 1: "女性"})
    
    # 创建节点和链接
    source = []
    target = []
    value = []
    
    # 性别 -> 年龄段
    for gender in sankey_df["性别_数值"].unique():
        for age in sankey_df["年龄段"].unique():
            count = len(sankey_df[(sankey_df["性别_数值"] == gender) & (sankey_df["年龄段"] == age)])
            if count > 0:
                source.append(gender)
                target.append(age)
                value.append(count)
    
    # 年龄段 -> 购买频率
    freq_labels = ["从不", "很少", "偶尔", "经常", "频繁"]
    for age in sankey_df["年龄段"].unique():
        for freq_val in sorted(sankey_df["购买频率_数值"].unique()):
            if int(freq_val) < len(freq_labels):
                freq = freq_labels[int(freq_val)]
                count = len(sankey_df[(sankey_df["年龄段"] == age) & (sankey_df["购买频率_数值"] == freq_val)])
                if count > 0:
                    source.append(age)
                    target.append(freq)
                    value.append(count)
    
    # 创建节点标签
    all_nodes = list(set(source + target))
    node_indices = {node: i for i, node in enumerate(all_nodes)}
    
    # 转换源和目标为索引
    source_indices = [node_indices[s] for s in source]
    target_indices = [node_indices[t] for t in target]
    
    # 创建桑基图
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_nodes,
            color=[px.colors.qualitative.Pastel[i % len(px.colors.qualitative.Pastel)] 
                  for i in range(len(all_nodes))]  # 使用不同颜色
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=value,
            color="rgba(100,100,100,0.2)"  # 半透明灰色连接
        ))])
    
    fig.update_layout(
        title_text="消费者特征流向图", 
        font_size=14,
        width=1000,
        height=800
    )
    
    fig.write_html("消费者特征流向图.html")
    print("已保存消费者特征流向图")

def perform_clustering_analysis(df):
    """
    使用K-means聚类分析消费者群体
    
    参数:
        df (pd.DataFrame): 数据框
    """
    print("\n===== 消费者聚类分析 =====")
    
    # 准备聚类数据
    cluster_data = df[["年龄", "购买频率_数值", "品牌认知度_数值"]].copy()
    
    # 标准化数据
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)
    
    # 确定最佳聚类数 (使用肘部法则)
    inertia = []
    k_range = range(1, 10)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(cluster_data_scaled)
        inertia.append(kmeans.inertia_)
    
    # 绘制肘部图
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'o-', color='purple')
    plt.xlabel('聚类数 (k)')
    plt.ylabel('惯性 (Inertia)')
    plt.title('K-means聚类肘部法则图')
    plt.grid(True)
    plt.savefig("聚类肘部法则图.png", dpi=300, bbox_inches="tight")
    print("已保存聚类肘部法则图")
    
    # 选择合适的聚类数 (假设为4)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42)
    df["聚类"] = kmeans.fit_predict(cluster_data_scaled)
    
    # 使用PCA降维以便可视化
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(cluster_data_scaled)
    df["PCA1"] = pca_result[:, 0]
    df["PCA2"] = pca_result[:, 1]
    
    # 可视化聚类结果
    plt.figure(figsize=(12, 10))
    
    # 散点图
    plt.subplot(2, 1, 1)
    sns.scatterplot(x="PCA1", y="PCA2", hue="聚类", data=df, palette="viridis", s=100, alpha=0.7)
    plt.title("消费者聚类分析 (PCA降维)", fontsize=14)
    plt.xlabel("主成分1")
    plt.ylabel("主成分2")
    
    # 聚类特征分析
    plt.subplot(2, 1, 2)
    cluster_profile = df.groupby("聚类")[["年龄", "购买频率_数值", "品牌认知度_数值"]].mean()
    
    # 标准化聚类特征以便比较
    cluster_profile_scaled = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())
    
    # 绘制热图
    sns.heatmap(cluster_profile_scaled, annot=cluster_profile.round(2), 
                cmap="YlGnBu", linewidths=.5, fmt=".2f")
    plt.title("各聚类群体特征分析", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("消费者聚类分析.png", dpi=300, bbox_inches="tight")
    print("已保存消费者聚类分析图")
    
    # 聚类解释
    print("\n消费者群体聚类解释:")
    for i in range(k):
        print(f"\n聚类 {i}:")
        print(f"  平均年龄: {cluster_profile.loc[i, '年龄']:.1f}")
        print(f"  平均购买频率: {cluster_profile.loc[i, '购买频率_数值']:.2f}")
        print(f"  平均品牌认知度: {cluster_profile.loc[i, '品牌认知度_数值']:.2f}")
        
        # 计算该聚类的性别分布
        gender_dist = df[df["聚类"] == i]["性别_数值"].map({0: "男", 1: "女"}).value_counts(normalize=True)
        print(f"  性别分布: 男 {gender_dist.get('男', 0)*100:.1f}%, 女 {gender_dist.get('女', 0)*100:.1f}%")
        
        # 计算该聚类的年龄段分布
        age_dist = df[df["聚类"] == i]["年龄段"].value_counts(normalize=True)
        if not age_dist.empty:
            print(f"  主要年龄段: {age_dist.index[0]} ({age_dist.values[0]*100:.1f}%)")
        else:
            print("  年龄段分布: 无数据")
        
        # 计算该聚类的购买频率分布
        freq_dist = df[df["聚类"] == i]["购买频率_数值"].value_counts(normalize=True)
        if not freq_dist.empty:
            freq_val = freq_dist.index[0]
            freq_labels = ["从不", "很少", "偶尔", "经常", "频繁"]
            if int(freq_val) < len(freq_labels):
                freq_label = freq_labels[int(freq_val)]
                print(f"  主要购买频率: {freq_label} ({freq_dist.values[0]*100:.1f}%)")
            else:
                print(f"  主要购买频率值: {freq_val} ({freq_dist.values[0]*100:.1f}%)")
        else:
            print("  购买频率分布: 无数据")
    
    # 创建交互式聚类可视化
    fig = px.scatter(df, x="PCA1", y="PCA2", color="聚类", 
                    hover_data=["年龄", "购买频率_数值", "品牌认知度_数值"],
                    title="消费者聚类交互式可视化")
    fig.write_html("消费者聚类交互式可视化.html")
    print("已保存消费者聚类交互式可视化")

def main():
    """主函数"""
    print("===== 狗牙儿零食品牌消费者画像分析 =====")
    
    # 设置文件路径
    file_path = "C:\\Users\\jiawang\\Desktop\\狗牙儿.xlsx"
    
    # 首先显示Excel文件的标题行
    df_original = show_excel_headers(file_path)
    
    if df_original is None:
        print("无法读取Excel文件，请检查文件路径和格式。")
        return
    
    # 加载数据
    df = load_data(file_path)
    
    # 基础消费者画像分析
    analyze_consumer_profile(df)
    
    # 高级可视化
    create_advanced_visualizations(df)
    
    # 聚类分析
    perform_clustering_analysis(df)
    
    print("\n分析完成！所有图表已保存。")

if __name__ == "__main__":
    main() 