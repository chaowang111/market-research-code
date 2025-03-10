"""
狗牙儿零食品牌产品偏好与营销分析
分析消费者的产品偏好、购买因素和营销渠道，为产品开发和营销策略提供数据支持

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
from wordcloud import WordCloud
import jieba
import os
import re
from collections import Counter
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
warnings.filterwarnings('ignore')

# 设置Plotly输出为浏览器
pio.renderers.default = "browser"

def load_data(file_path):
    """
    加载Excel数据并进行预处理
    
    参数:
        file_path (str): Excel文件路径
        
    返回:
        pd.DataFrame: 处理后的数据框
    """
    print(f"正在加载数据: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    
    # 读取Excel文件
    df = pd.read_excel(file_path)
    print(f"数据加载完成，共 {len(df)} 行记录")
    
    # 显示所有列名
    print("\n数据列名:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    return df

def analyze_purchase_factors(df):
    """
    分析消费者购买因素
    
    参数:
        df (pd.DataFrame): 数据框
    """
    print("\n===== 购买因素分析 =====")
    
    # 查找购买因素相关的列
    purchase_factor_cols = []
    for col in df.columns:
        if '购买' in str(col) and '因素' in str(col) or '因什么购买' in str(col):
            purchase_factor_cols.append(col)
    
    if not purchase_factor_cols:
        print("未找到购买因素相关的列")
        return
    
    print(f"找到以下购买因素相关的列: {purchase_factor_cols}")
    
    # 选择第一个购买因素列进行分析
    factor_col = purchase_factor_cols[0]
    print(f"\n分析列: {factor_col}")
    
    # 处理多选题数据
    all_factors = []
    for response in df[factor_col].dropna():
        # 尝试不同的分隔符
        if isinstance(response, str):
            if ',' in response or '，' in response:
                factors = re.split(r'[,，]', response)
            elif ';' in response or '；' in response:
                factors = re.split(r'[;；]', response)
            elif '、' in response:
                factors = response.split('、')
            else:
                factors = [response]
            
            all_factors.extend([f.strip() for f in factors if f.strip()])
    
    # 统计各因素出现频率
    factor_counts = Counter(all_factors)
    
    # 转换为DataFrame
    factor_df = pd.DataFrame({
        '购买因素': list(factor_counts.keys()),
        '频次': list(factor_counts.values())
    }).sort_values('频次', ascending=False)
    
    # 只保留前10个因素
    if len(factor_df) > 10:
        factor_df = factor_df.head(10)
    
    # 创建条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='频次', y='购买因素', data=factor_df, palette='viridis')
    plt.title(f"消费者购买狗牙儿产品的主要因素 (Top {len(factor_df)})", fontsize=16)
    plt.xlabel('提及频次', fontsize=14)
    plt.ylabel('购买因素', fontsize=14)
    plt.tight_layout()
    plt.savefig("购买因素分析.png", dpi=300, bbox_inches="tight")
    print("已保存购买因素分析图表")
    
    # 创建词云
    if all_factors:
        text = ' '.join(all_factors)
        wordcloud = WordCloud(
            font_path='simhei.ttf',  # 使用中文字体
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(text)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("购买因素词云", fontsize=16)
        plt.tight_layout()
        plt.savefig("购买因素词云.png", dpi=300, bbox_inches="tight")
        print("已保存购买因素词云图")
    
    # 创建交互式条形图
    fig = px.bar(
        factor_df, 
        x='频次', 
        y='购买因素',
        orientation='h',
        color='频次',
        color_continuous_scale='Viridis',
        title=f"消费者购买狗牙儿产品的主要因素 (Top {len(factor_df)})"
    )
    
    fig.update_layout(
        xaxis_title="提及频次",
        yaxis_title="购买因素",
        font=dict(size=14),
        height=600,
        width=900
    )
    
    fig.write_html("购买因素分析.html")
    print("已保存购买因素交互式图表")

def analyze_product_preferences(df):
    """
    分析消费者产品偏好
    
    参数:
        df (pd.DataFrame): 数据框
    """
    print("\n===== 产品偏好分析 =====")
    
    # 查找产品偏好相关的列
    product_cols = []
    flavor_cols = []
    
    for col in df.columns:
        col_str = str(col).lower()
        if '产品' in col_str or '食用过' in col_str:
            product_cols.append(col)
        elif '口味' in col_str:
            flavor_cols.append(col)
    
    # 分析产品偏好
    if product_cols:
        print(f"找到以下产品相关的列: {product_cols}")
        product_col = product_cols[0]
        print(f"\n分析列: {product_col}")
        
        # 处理多选题数据
        all_products = []
        for response in df[product_col].dropna():
            if isinstance(response, str):
                if ',' in response or '，' in response:
                    products = re.split(r'[,，]', response)
                elif ';' in response or '；' in response:
                    products = re.split(r'[;；]', response)
                elif '、' in response:
                    products = response.split('、')
                else:
                    products = [response]
                
                all_products.extend([p.strip() for p in products if p.strip()])
        
        # 统计各产品出现频率
        product_counts = Counter(all_products)
        
        # 转换为DataFrame
        product_df = pd.DataFrame({
            '产品': list(product_counts.keys()),
            '频次': list(product_counts.values())
        }).sort_values('频次', ascending=False)
        
        # 只保留前10个产品
        if len(product_df) > 10:
            product_df = product_df.head(10)
        
        # 创建条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='频次', y='产品', data=product_df, palette='magma')
        plt.title(f"消费者最常购买的狗牙儿产品 (Top {len(product_df)})", fontsize=16)
        plt.xlabel('提及频次', fontsize=14)
        plt.ylabel('产品', fontsize=14)
        plt.tight_layout()
        plt.savefig("产品偏好分析.png", dpi=300, bbox_inches="tight")
        print("已保存产品偏好分析图表")
        
        # 创建交互式条形图
        fig = px.bar(
            product_df, 
            x='频次', 
            y='产品',
            orientation='h',
            color='频次',
            color_continuous_scale='Magma',
            title=f"消费者最常购买的狗牙儿产品 (Top {len(product_df)})"
        )
        
        fig.update_layout(
            xaxis_title="提及频次",
            yaxis_title="产品",
            font=dict(size=14),
            height=600,
            width=900
        )
        
        fig.write_html("产品偏好分析.html")
        print("已保存产品偏好交互式图表")
    else:
        print("未找到产品相关的列")
    
    # 分析口味偏好
    if flavor_cols:
        print(f"\n找到以下口味相关的列: {flavor_cols}")
        flavor_col = flavor_cols[0]
        print(f"分析列: {flavor_col}")
        
        # 处理多选题数据
        all_flavors = []
        for response in df[flavor_col].dropna():
            if isinstance(response, str):
                if ',' in response or '，' in response:
                    flavors = re.split(r'[,，]', response)
                elif ';' in response or '；' in response:
                    flavors = re.split(r'[;；]', response)
                elif '、' in response:
                    flavors = response.split('、')
                else:
                    flavors = [response]
                
                all_flavors.extend([f.strip() for f in flavors if f.strip()])
        
        # 统计各口味出现频率
        flavor_counts = Counter(all_flavors)
        
        # 转换为DataFrame
        flavor_df = pd.DataFrame({
            '口味': list(flavor_counts.keys()),
            '频次': list(flavor_counts.values())
        }).sort_values('频次', ascending=False)
        
        # 只保留前10个口味
        if len(flavor_df) > 10:
            flavor_df = flavor_df.head(10)
        
        # 创建条形图
        plt.figure(figsize=(12, 8))
        sns.barplot(x='频次', y='口味', data=flavor_df, palette='rocket')
        plt.title(f"消费者最喜欢的狗牙儿产品口味 (Top {len(flavor_df)})", fontsize=16)
        plt.xlabel('提及频次', fontsize=14)
        plt.ylabel('口味', fontsize=14)
        plt.tight_layout()
        plt.savefig("口味偏好分析.png", dpi=300, bbox_inches="tight")
        print("已保存口味偏好分析图表")
        
        # 创建饼图
        plt.figure(figsize=(12, 10))
        plt.pie(
            flavor_df['频次'], 
            labels=flavor_df['口味'], 
            autopct='%1.1f%%',
            startangle=90, 
            shadow=True,
            colors=sns.color_palette('rocket', len(flavor_df))
        )
        plt.axis('equal')
        plt.title("消费者口味偏好分布", fontsize=16)
        plt.tight_layout()
        plt.savefig("口味偏好饼图.png", dpi=300, bbox_inches="tight")
        print("已保存口味偏好饼图")
        
        # 创建交互式饼图
        fig = px.pie(
            flavor_df, 
            values='频次', 
            names='口味',
            title="消费者口味偏好分布",
            color_discrete_sequence=px.colors.sequential.Plasma_r
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(size=14),
            height=700,
            width=900
        )
        
        fig.write_html("口味偏好分析.html")
        print("已保存口味偏好交互式图表")
    else:
        print("未找到口味相关的列")

def analyze_marketing_channels(df):
    """
    分析营销渠道效果
    
    参数:
        df (pd.DataFrame): 数据框
    """
    print("\n===== 营销渠道分析 =====")
    
    # 查找营销渠道相关的列
    channel_cols = []
    for col in df.columns:
        col_str = str(col).lower()
        if '渠道' in col_str or '了解' in col_str and '狗牙儿' in col_str:
            channel_cols.append(col)
    
    if not channel_cols:
        print("未找到营销渠道相关的列")
        return
    
    print(f"找到以下营销渠道相关的列: {channel_cols}")
    
    # 选择第一个渠道列进行分析
    channel_col = channel_cols[0]
    print(f"\n分析列: {channel_col}")
    
    # 处理多选题数据
    all_channels = []
    for response in df[channel_col].dropna():
        if isinstance(response, str):
            if ',' in response or '，' in response:
                channels = re.split(r'[,，]', response)
            elif ';' in response or '；' in response:
                channels = re.split(r'[;；]', response)
            elif '、' in response:
                channels = response.split('、')
            else:
                channels = [response]
            
            all_channels.extend([c.strip() for c in channels if c.strip()])
    
    # 统计各渠道出现频率
    channel_counts = Counter(all_channels)
    
    # 转换为DataFrame
    channel_df = pd.DataFrame({
        '营销渠道': list(channel_counts.keys()),
        '频次': list(channel_counts.values())
    }).sort_values('频次', ascending=False)
    
    # 创建条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='频次', y='营销渠道', data=channel_df, palette='mako')
    plt.title("消费者了解狗牙儿品牌的主要渠道", fontsize=16)
    plt.xlabel('提及频次', fontsize=14)
    plt.ylabel('营销渠道', fontsize=14)
    plt.tight_layout()
    plt.savefig("营销渠道分析.png", dpi=300, bbox_inches="tight")
    print("已保存营销渠道分析图表")
    
    # 创建交互式条形图
    fig = px.bar(
        channel_df, 
        x='频次', 
        y='营销渠道',
        orientation='h',
        color='频次',
        color_continuous_scale='Mako',
        title="消费者了解狗牙儿品牌的主要渠道"
    )
    
    fig.update_layout(
        xaxis_title="提及频次",
        yaxis_title="营销渠道",
        font=dict(size=14),
        height=600,
        width=900
    )
    
    fig.write_html("营销渠道分析.html")
    print("已保存营销渠道交互式图表")
    
    # 创建饼图
    plt.figure(figsize=(12, 10))
    plt.pie(
        channel_df['频次'], 
        labels=channel_df['营销渠道'], 
        autopct='%1.1f%%',
        startangle=90, 
        shadow=True,
        colors=sns.color_palette('mako', len(channel_df))
    )
    plt.axis('equal')
    plt.title("营销渠道分布", fontsize=16)
    plt.tight_layout()
    plt.savefig("营销渠道饼图.png", dpi=300, bbox_inches="tight")
    print("已保存营销渠道饼图")

def analyze_promotion_effectiveness(df):
    """
    分析促销活动效果
    
    参数:
        df (pd.DataFrame): 数据框
    """
    print("\n===== 促销活动效果分析 =====")
    
    # 查找促销活动相关的列
    promotion_cols = []
    for col in df.columns:
        col_str = str(col).lower()
        if '促销' in col_str or '活动' in col_str or '购买意愿' in col_str:
            promotion_cols.append(col)
    
    if not promotion_cols:
        print("未找到促销活动相关的列")
        return
    
    print(f"找到以下促销活动相关的列: {promotion_cols}")
    
    # 选择第一个促销列进行分析
    promotion_col = promotion_cols[0]
    print(f"\n分析列: {promotion_col}")
    
    # 处理多选题数据
    all_promotions = []
    for response in df[promotion_col].dropna():
        if isinstance(response, str):
            if ',' in response or '，' in response:
                promotions = re.split(r'[,，]', response)
            elif ';' in response or '；' in response:
                promotions = re.split(r'[;；]', response)
            elif '、' in response:
                promotions = response.split('、')
            else:
                promotions = [response]
            
            all_promotions.extend([p.strip() for p in promotions if p.strip()])
    
    # 统计各促销活动出现频率
    promotion_counts = Counter(all_promotions)
    
    # 转换为DataFrame
    promotion_df = pd.DataFrame({
        '促销活动': list(promotion_counts.keys()),
        '频次': list(promotion_counts.values())
    }).sort_values('频次', ascending=False)
    
    # 创建条形图
    plt.figure(figsize=(12, 8))
    sns.barplot(x='频次', y='促销活动', data=promotion_df, palette='flare')
    plt.title("最能增加消费者购买意愿的促销活动", fontsize=16)
    plt.xlabel('提及频次', fontsize=14)
    plt.ylabel('促销活动', fontsize=14)
    plt.tight_layout()
    plt.savefig("促销活动效果分析.png", dpi=300, bbox_inches="tight")
    print("已保存促销活动效果分析图表")
    
    # 创建交互式条形图
    fig = px.bar(
        promotion_df, 
        x='频次', 
        y='促销活动',
        orientation='h',
        color='频次',
        color_continuous_scale='Inferno',
        title="最能增加消费者购买意愿的促销活动"
    )
    
    fig.update_layout(
        xaxis_title="提及频次",
        yaxis_title="促销活动",
        font=dict(size=14),
        height=600,
        width=900
    )
    
    fig.write_html("促销活动效果分析.html")
    print("已保存促销活动效果交互式图表")

def main():
    """主函数"""
    print("===== 狗牙儿零食品牌产品偏好与营销分析 =====")
    
    # 设置文件路径
    file_path = "C:\\Users\\jiawang\\Desktop\\狗牙儿.xlsx"
    
    # 加载数据
    df = load_data(file_path)
    
    if df is None:
        print("无法读取Excel文件，请检查文件路径和格式。")
        return
    
    # 创建输出目录
    output_dir = "产品偏好与营销分析"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    # 切换到输出目录
    os.chdir(output_dir)
    
    # 分析购买因素
    analyze_purchase_factors(df)
    
    # 分析产品偏好
    analyze_product_preferences(df)
    
    # 分析营销渠道
    analyze_marketing_channels(df)
    
    # 分析促销活动效果
    analyze_promotion_effectiveness(df)
    
    # 返回上级目录
    os.chdir("..")
    
    print("\n分析完成！所有图表已保存到 '产品偏好与营销分析' 目录。")

if __name__ == "__main__":
    main() 