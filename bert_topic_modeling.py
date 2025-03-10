import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from transformers import BertModel, BertTokenizer
import umap
import hdbscan
from collections import Counter
import jieba
import re
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from pathlib import Path
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_excel_data(file_path):
    """加载Excel数据"""
    print(f"正在从{file_path}加载数据...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        # 尝试在当前目录查找Excel文件
        excel_files = list(Path('.').glob('*.xlsx'))
        if excel_files:
            file_path = str(excel_files[0])
            print(f"使用找到的Excel文件: {file_path}")
        else:
            raise FileNotFoundError(f"找不到Excel文件")
    
    # 加载数据
    df = pd.read_excel(file_path)
    print(f"成功加载数据，共{len(df)}条记录")
    
    return df

def preprocess_text(text):
    """预处理文本"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # 移除标点符号和特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    
    # 分词
    words = jieba.cut(text)
    
    # 移除停用词（可以扩展）
    stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '可以', '没', '啊', '吧', '呢', '啥', '那', '这个', '那个', '什么', '怎么', '如何', '为什么', '多少', '哪里', '哪个', '哪些', '谁', '什么时候', '怎样', '为何', '多久', '多长时间', '多远', '多高', '多大', '多宽', '多深', '多重', '多少钱', '多少人', '多少次', '多少个', '多少种', '多少家', '多少岁', '多少年', '多少月', '多少日', '多少天', '多少周', '多少小时', '多少分钟', '多少秒'}
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    
    return ' '.join(filtered_words)

def extract_bert_embeddings(texts, batch_size=8):
    """提取BERT嵌入向量"""
    print("正在提取BERT嵌入向量...")
    
    # 设置环境变量，使用国内镜像
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 加载预训练的BERT模型和分词器
    try:
        print("尝试从Hugging Face下载模型...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese')
    except Exception as e:
        print(f"从Hugging Face下载模型出错: {e}")
        
        try:
            print("尝试从本地缓存加载...")
            # 尝试从缓存加载
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "transformers")
            if os.path.exists(cache_dir):
                print(f"使用缓存目录: {cache_dir}")
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir=cache_dir)
                model = BertModel.from_pretrained('bert-base-chinese', cache_dir=cache_dir)
            else:
                print("缓存目录不存在，尝试使用离线模式...")
                # 尝试使用离线模式
                tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
                model = BertModel.from_pretrained('bert-base-chinese', local_files_only=True)
        except Exception as e2:
            print(f"从本地加载模型出错: {e2}")
            print("尝试使用替代模型...")
            
            try:
                # 尝试使用其他中文模型
                print("尝试加载chinese-bert-wwm模型...")
                tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
                model = BertModel.from_pretrained('hfl/chinese-bert-wwm-ext')
            except Exception as e3:
                print(f"加载替代模型出错: {e3}")
                raise Exception("无法加载任何BERT模型，请检查网络连接或手动下载模型")
    
    print("BERT模型加载成功!")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = model.to(device)
    model.eval()
    
    # 分批处理
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        print(f"处理批次 {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}...")
        
        # 对文本进行编码
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # 提取嵌入向量
        with torch.no_grad():
            output = model(**encoded_input)
        
        # 使用[CLS]标记的嵌入作为文本表示
        batch_embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(batch_embeddings)
    
    # 合并所有批次的嵌入向量
    embeddings = np.vstack(embeddings)
    print(f"成功提取 {len(embeddings)} 个嵌入向量，维度: {embeddings.shape[1]}")
    
    return embeddings

def reduce_dimensions(embeddings, n_neighbors=15, min_dist=0.1, n_components=2):
    """使用UMAP降维"""
    print("正在使用UMAP进行降维...")
    
    # 创建UMAP模型
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='cosine',
        random_state=42
    )
    
    # 降维
    reduced_embeddings = reducer.fit_transform(embeddings)
    print(f"降维完成，新维度: {reduced_embeddings.shape[1]}")
    
    return reduced_embeddings

def cluster_texts(reduced_embeddings, min_cluster_size=5):
    """使用HDBSCAN进行聚类"""
    print("正在使用HDBSCAN进行聚类...")
    
    # 创建HDBSCAN模型
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=1,
        metric='euclidean',
        cluster_selection_epsilon=0.5,
        prediction_data=True
    )
    
    # 聚类
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    # 统计每个簇的数量
    cluster_counts = Counter(cluster_labels)
    print(f"聚类完成，共有 {len(cluster_counts)} 个簇")
    for cluster_id, count in sorted(cluster_counts.items()):
        if cluster_id == -1:
            print(f"  噪声点: {count} 个")
        else:
            print(f"  簇 {cluster_id}: {count} 个")
    
    return cluster_labels

def find_optimal_k(reduced_embeddings, max_k=10):
    """寻找最佳的K值"""
    print("正在寻找最佳的K值...")
    
    # 计算不同K值的轮廓系数
    silhouette_scores = []
    k_values = range(2, min(max_k + 1, len(reduced_embeddings)))
    
    for k in k_values:
        # 创建KMeans模型
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        
        # 聚类
        cluster_labels = kmeans.fit_predict(reduced_embeddings)
        
        # 计算轮廓系数
        score = silhouette_score(reduced_embeddings, cluster_labels)
        silhouette_scores.append(score)
        print(f"  K = {k}: 轮廓系数 = {score:.4f}")
    
    # 找到最佳的K值
    best_k = k_values[np.argmax(silhouette_scores)]
    print(f"最佳的K值: {best_k}")
    
    return best_k

def kmeans_clustering(reduced_embeddings, k):
    """使用KMeans进行聚类"""
    print(f"正在使用KMeans进行聚类 (K = {k})...")
    
    # 创建KMeans模型
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    
    # 聚类
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    
    # 统计每个簇的数量
    cluster_counts = Counter(cluster_labels)
    print(f"聚类完成，共有 {len(cluster_counts)} 个簇")
    for cluster_id, count in sorted(cluster_counts.items()):
        print(f"  簇 {cluster_id}: {count} 个")
    
    return cluster_labels

def extract_keywords_per_cluster(texts, cluster_labels, top_n=10):
    """提取每个簇的关键词"""
    print("正在提取每个簇的关键词...")
    
    # 获取所有簇的ID
    unique_clusters = sorted(set(cluster_labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # 移除噪声点
    
    # 为每个簇提取关键词
    keywords_per_cluster = {}
    for cluster_id in unique_clusters:
        # 获取该簇的所有文本
        cluster_texts = [text for i, text in enumerate(texts) if cluster_labels[i] == cluster_id]
        
        # 将文本合并为一个大文本
        combined_text = ' '.join(cluster_texts)
        
        # 分词并统计词频
        words = combined_text.split()
        word_counts = Counter(words)
        
        # 获取出现频率最高的词
        top_words = word_counts.most_common(top_n)
        
        # 存储关键词
        keywords_per_cluster[cluster_id] = [word for word, count in top_words]
    
    return keywords_per_cluster

def visualize_clusters(reduced_embeddings, cluster_labels, texts, keywords_per_cluster, output_dir="文本主题分析"):
    """可视化聚类结果"""
    print("正在可视化聚类结果...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有簇的ID
    unique_clusters = sorted(set(cluster_labels))
    
    # 为每个簇分配不同的颜色
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    color_map = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
    
    # 创建散点图
    plt.figure(figsize=(12, 8))
    
    # 绘制每个簇
    for cluster_id in unique_clusters:
        # 获取该簇的所有点
        mask = cluster_labels == cluster_id
        points = reduced_embeddings[mask]
        
        # 绘制散点
        if cluster_id == -1:
            # 噪声点
            plt.scatter(points[:, 0], points[:, 1], c='gray', label='噪声', alpha=0.5, s=30)
        else:
            # 正常簇
            plt.scatter(points[:, 0], points[:, 1], c=[color_map[cluster_id]], label=f'簇 {cluster_id}', alpha=0.8, s=50)
            
            # 计算簇的中心点
            centroid = points.mean(axis=0)
            
            # 添加簇的标签
            keywords = keywords_per_cluster.get(cluster_id, [])
            if keywords:
                # 使用前3个关键词作为标签
                label = ', '.join(keywords[:3])
                plt.annotate(label, centroid, fontsize=12, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    # 添加标题和图例
    plt.title('文本主题聚类可视化', fontsize=16)
    plt.xlabel('UMAP维度1', fontsize=12)
    plt.ylabel('UMAP维度2', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_clusters.png'), dpi=300, bbox_inches='tight')
    
    # 为每个簇创建词云
    print("正在生成每个簇的词云...")
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue  # 跳过噪声点
        
        # 获取该簇的所有文本
        cluster_texts = [text for i, text in enumerate(texts) if cluster_labels[i] == cluster_id]
        
        # 将文本合并为一个大文本
        combined_text = ' '.join(cluster_texts)
        
        # 创建词云
        try:
            font_path = 'C:\\Windows\\Fonts\\simhei.ttf'  # Windows系统中文字体路径
            if not os.path.exists(font_path):
                # 尝试查找系统中的中文字体
                font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
                chinese_fonts = [f for f in font_paths if os.path.basename(f).startswith(('sim', 'msyh', 'Sim', 'MSYH'))]
                if chinese_fonts:
                    font_path = chinese_fonts[0]
                else:
                    font_path = None
            
            if font_path:
                wordcloud = WordCloud(
                    font_path=font_path,
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    max_font_size=100,
                    random_state=42
                ).generate(combined_text)
            else:
                # 如果找不到中文字体，使用默认字体
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    max_font_size=100,
                    random_state=42
                ).generate(combined_text)
            
            # 创建图像
            plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(f'簇 {cluster_id} 的词云 - {", ".join(keywords_per_cluster.get(cluster_id, [])[:5])}', fontsize=14)
            
            # 保存图像
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'wordcloud_cluster_{cluster_id}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"生成簇 {cluster_id} 的词云时出错: {e}")
    
    # 创建热图显示每个簇的关键词分布
    print("正在生成关键词热图...")
    try:
        # 获取所有簇的关键词
        all_keywords = []
        for keywords in keywords_per_cluster.values():
            all_keywords.extend(keywords)
        unique_keywords = sorted(set(all_keywords))
        
        # 创建热图数据
        heatmap_data = np.zeros((len(keywords_per_cluster), len(unique_keywords)))
        cluster_ids = sorted(keywords_per_cluster.keys())
        
        for i, cluster_id in enumerate(cluster_ids):
            keywords = keywords_per_cluster[cluster_id]
            for keyword in keywords:
                if keyword in unique_keywords:
                    j = unique_keywords.index(keyword)
                    heatmap_data[i, j] = 1
        
        # 创建热图
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_data, cmap='YlGnBu', 
                   xticklabels=unique_keywords, 
                   yticklabels=[f'簇 {cluster_id}' for cluster_id in cluster_ids])
        plt.title('各簇关键词分布热图', fontsize=16)
        plt.xlabel('关键词', fontsize=12)
        plt.ylabel('簇', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'keyword_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"生成关键词热图时出错: {e}")
    
    print(f"可视化结果已保存到 '{output_dir}' 目录")

def print_cluster_summary(texts, cluster_labels, keywords_per_cluster, output_dir="文本主题分析"):
    """打印每个簇的摘要"""
    print("\n=== 主题聚类摘要 ===")
    
    # 获取所有簇的ID
    unique_clusters = sorted(set(cluster_labels))
    if -1 in unique_clusters:
        unique_clusters.remove(-1)  # 移除噪声点
    
    # 创建摘要文件
    summary_file = os.path.join(output_dir, 'topic_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=== 狗牙儿产品建议与期望 - 主题聚类摘要 ===\n\n")
        
        # 为每个簇打印摘要
        for cluster_id in unique_clusters:
            # 获取该簇的所有文本
            cluster_texts = [text for i, text in enumerate(texts) if cluster_labels[i] == cluster_id]
            
            # 获取关键词
            keywords = keywords_per_cluster.get(cluster_id, [])
            
            # 打印簇的信息
            cluster_info = f"主题 {cluster_id}: {', '.join(keywords[:5])} (共 {len(cluster_texts)} 条)"
            print(cluster_info)
            f.write(f"{cluster_info}\n")
            
            # 打印示例文本
            print("示例文本:")
            f.write("示例文本:\n")
            for i, text in enumerate(cluster_texts[:5]):  # 只显示前5个示例
                print(f"  {i+1}. {text}")
                f.write(f"  {i+1}. {text}\n")
            
            print()
            f.write("\n")
    
    print(f"摘要已保存到 '{summary_file}'")

def compare_with_lda(texts, n_topics=5, output_dir="文本主题分析"):
    """与传统LDA主题模型进行比较"""
    print("\n=== 与传统LDA主题模型比较 ===")
    
    # 创建词袋模型
    vectorizer = CountVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # 创建LDA模型
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    # 获取特征名称
    feature_names = vectorizer.get_feature_names_out()
    
    # 打印每个主题的关键词
    lda_keywords = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        lda_keywords[topic_idx] = top_words
        print(f"LDA主题 {topic_idx}: {', '.join(top_words)}")
    
    # 保存LDA结果
    lda_file = os.path.join(output_dir, 'lda_topics.txt')
    with open(lda_file, 'w', encoding='utf-8') as f:
        f.write("=== 传统LDA主题模型结果 ===\n\n")
        for topic_idx, keywords in lda_keywords.items():
            f.write(f"主题 {topic_idx}: {', '.join(keywords)}\n")
    
    print(f"LDA结果已保存到 '{lda_file}'")
    
    return lda_keywords

def main():
    """主函数"""
    # 设置输出目录
    output_dir = "文本主题分析"
    os.makedirs(output_dir, exist_ok=True)
    
    # 尝试加载数据
    try:
        # 首先尝试从默认路径加载
        excel_path = "C:\\Users\\jiawang\\Desktop\\狗牙儿.xlsx"
        if not os.path.exists(excel_path):
            # 如果默认路径不存在，尝试在当前目录查找Excel文件
            excel_files = list(Path('.').glob('*.xlsx'))
            if excel_files:
                excel_path = str(excel_files[0])
            else:
                raise FileNotFoundError("找不到Excel文件")
        
        df = load_excel_data(excel_path)
    except Exception as e:
        print(f"加载数据出错: {e}")
        return
    
    # 提取问题22的回答
    question_column = '22. 您对狗牙儿产品有什么建议或期望？'
    
    # 检查列是否存在
    if question_column not in df.columns:
        # 尝试查找类似的列
        similar_columns = [col for col in df.columns if '建议' in col or '期望' in col or '改进' in col]
        if similar_columns:
            question_column = similar_columns[0]
            print(f"使用列: {question_column}")
        else:
            print("找不到包含建议或期望的列")
            return
    
    # 提取文本
    texts = df[question_column].dropna().tolist()
    
    # 检查文本数量
    if len(texts) < 10:
        print("文本数量太少，无法进行有效的主题建模")
        return
    
    print(f"共有 {len(texts)} 条文本回答")
    
    # 预处理文本
    preprocessed_texts = [preprocess_text(text) for text in texts]
    
    # 过滤掉空文本
    valid_indices = [i for i, text in enumerate(preprocessed_texts) if text]
    filtered_texts = [preprocessed_texts[i] for i in valid_indices]
    original_texts = [texts[i] for i in valid_indices]
    
    print(f"预处理后剩余 {len(filtered_texts)} 条有效文本")
    
    # 选择分析方法
    analysis_method = "bert"  # 默认使用BERT
    
    # 提取BERT嵌入向量
    try:
        if analysis_method == "bert":
            print("使用BERT进行主题建模...")
            embeddings = extract_bert_embeddings(filtered_texts)
            
            # 降维
            reduced_embeddings = reduce_dimensions(embeddings)
            
            # 尝试HDBSCAN聚类
            try:
                cluster_labels = cluster_texts(reduced_embeddings)
                
                # 检查是否有足够的簇
                if len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) < 2:
                    print("HDBSCAN聚类结果不理想，尝试KMeans聚类")
                    raise Exception("簇数量不足")
            except Exception as e:
                print(f"HDBSCAN聚类出错: {e}")
                
                # 寻找最佳的K值
                best_k = find_optimal_k(reduced_embeddings)
                
                # 使用KMeans聚类
                cluster_labels = kmeans_clustering(reduced_embeddings, best_k)
            
            # 提取每个簇的关键词
            keywords_per_cluster = extract_keywords_per_cluster(filtered_texts, cluster_labels)
            
            # 打印每个簇的摘要
            print_cluster_summary(original_texts, cluster_labels, keywords_per_cluster, output_dir=output_dir)
            
            # 可视化聚类结果
            visualize_clusters(reduced_embeddings, cluster_labels, filtered_texts, keywords_per_cluster, output_dir=output_dir)
            
            # 与传统LDA主题模型比较
            lda_keywords = compare_with_lda(filtered_texts, n_topics=len(keywords_per_cluster), output_dir=output_dir)
            
            print("\nBERT主题建模分析完成！所有结果已保存到 '文本主题分析' 目录")
        
    except Exception as e:
        print(f"BERT分析出错: {e}")
        print("切换到传统LDA主题模型...")
        
        # 使用传统LDA主题模型
        try:
            # 确定主题数量
            n_topics = 5  # 默认值
            
            # 尝试找到最佳主题数
            print("尝试确定最佳主题数...")
            from sklearn.model_selection import GridSearchCV
            from sklearn.decomposition import LatentDirichletAllocation
            
            # 创建词袋模型
            vectorizer = CountVectorizer(max_features=1000)
            X = vectorizer.fit_transform(filtered_texts)
            
            # 尝试不同的主题数
            try:
                search_params = {'n_components': [3, 4, 5, 6, 7, 8]}
                lda = LatentDirichletAllocation()
                model = GridSearchCV(lda, param_grid=search_params)
                model.fit(X)
                
                # 获取最佳主题数
                best_lda_model = model.best_estimator_
                n_topics = best_lda_model.n_components
                print(f"最佳主题数: {n_topics}")
                
                # 使用最佳模型
                lda_keywords = compare_with_lda(filtered_texts, n_topics=n_topics, output_dir=output_dir)
            except Exception as e:
                print(f"自动确定主题数失败: {e}")
                # 使用默认主题数
                lda_keywords = compare_with_lda(filtered_texts, n_topics=n_topics, output_dir=output_dir)
            
            print("\nLDA主题建模分析完成！结果已保存到 '文本主题分析' 目录")
            
        except Exception as e:
            print(f"LDA分析也失败了: {e}")
            print("无法完成主题建模分析")

if __name__ == "__main__":
    main()
