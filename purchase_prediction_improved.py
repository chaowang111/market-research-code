import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import shap
import os
import re
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_excel_data(file_path):
    """加载Excel数据"""
    print(f"正在从{file_path}加载数据...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载数据
    df = pd.read_excel(file_path)
    print(f"成功加载数据，共{len(df)}条记录")
    
    return df

def extract_number(value):
    """从字符串中提取数字"""
    if pd.isna(value):
        return np.nan
    
    # 将值转换为字符串
    value_str = str(value)
    
    # 尝试直接转换（如果已经是数字）
    try:
        # 移除百分号
        value_str = value_str.replace('%', '')
        return float(value_str)
    except ValueError:
        pass
    
    # 尝试提取数字
    # 匹配模式：数字（可能有小数点）后面可能跟着"以内"、"以下"等
    match = re.search(r'(\d+\.?\d*)', value_str)
    if match:
        return float(match.group(1))
    
    # 处理特殊情况
    if '不接受' in value_str or '不会' in value_str:
        return 0.0
    
    # 如果都不匹配，返回NaN
    return np.nan

def preprocess_data(df):
    """预处理数据"""
    print("正在预处理数据...")
    
    # 创建新的DataFrame用于存储处理后的数据
    data = pd.DataFrame()
    
    # 提取购买频率并转换为数值
    if '3. 您购买休闲零食的频率:' in df.columns:
        frequency_map = {
            '每天': 5,
            '每周数次': 4,
            '每周一次': 3,
            '每月数次': 2,
            '偶尔': 1,
            '从不': 0
        }
        data['购买频率'] = df['3. 您购买休闲零食的频率:'].map(frequency_map)
    
    # 处理多选题（购买动机）
    if '7. 您会因什么购买狗牙儿产品?（可多选）' in df.columns:
        # 确保是字符串类型并处理缺失值
        df['7. 您会因什么购买狗牙儿产品?（可多选）'] = df['7. 您会因什么购买狗牙儿产品?（可多选）'].fillna('')
        
        # 提取所有可能的动机
        all_motives = []
        for motives in df['7. 您会因什么购买狗牙儿产品?（可多选）']:
            if isinstance(motives, str):
                motives_list = [m.strip() for m in motives.split(',')]
                all_motives.extend(motives_list)
        
        # 创建动机特征
        unique_motives = set([m for m in all_motives if m])
        for motive in unique_motives:
            data[f'动机_{motive}'] = df['7. 您会因什么购买狗牙儿产品?（可多选）'].apply(
                lambda x: 1 if isinstance(x, str) and motive in x else 0
            )
    
    # 处理多选题（口味偏好）
    if '8. 您喜欢狗牙儿食品的哪种口味?（可多选）' in df.columns:
        # 确保是字符串类型并处理缺失值
        df['8. 您喜欢狗牙儿食品的哪种口味?（可多选）'] = df['8. 您喜欢狗牙儿食品的哪种口味?（可多选）'].fillna('')
        
        # 提取所有可能的口味
        all_flavors = []
        for flavors in df['8. 您喜欢狗牙儿食品的哪种口味?（可多选）']:
            if isinstance(flavors, str):
                flavors_list = [f.strip() for f in flavors.split(',')]
                all_flavors.extend(flavors_list)
        
        # 创建口味特征
        unique_flavors = set([f for f in all_flavors if f])
        for flavor in unique_flavors:
            data[f'口味_{flavor}'] = df['8. 您喜欢狗牙儿食品的哪种口味?（可多选）'].apply(
                lambda x: 1 if isinstance(x, str) and flavor in x else 0
            )
    
    # 处理健康关注度
    if '13. 您是否关注零食的健康成分（如低油、低盐、无添加）?' in df.columns:
        health_concern_map = {
            '非常关注': 4,
            '比较关注': 3,
            '一般': 2,
            '不太关注': 1,
            '完全不关注': 0
        }
        data['健康关注度'] = df['13. 您是否关注零食的健康成分（如低油、低盐、无添加）?'].map(health_concern_map)
    
    # 处理健康版购买意愿
    if '14. 您是否会因狗牙儿推出"健康版"产品（如非油炸、全谷物）而增加购买意愿?' in df.columns:
        intention_map = {
            '一定会': 4,
            '可能会': 3,
            '不确定': 2,
            '可能不会': 1,
            '一定不会': 0
        }
        data['健康版购买意愿'] = df['14. 您是否会因狗牙儿推出"健康版"产品（如非油炸、全谷物）而增加购买意愿?'].map(intention_map)
    
    # 处理健康版价格承受度（使用正则表达式提取数字）
    if '15. 您能接受健康化产品的价格比普通款高多少?' in df.columns:
        data['健康版价格承受度'] = df['15. 您能接受健康化产品的价格比普通款高多少?'].apply(extract_number)
    
    # 处理备选列（如果主列不存在）
    if '健康版价格承受度' not in data.columns and '30.您能接受健康化产品的价格比普通款高多少？' in df.columns:
        data['健康版价格承受度'] = df['30.您能接受健康化产品的价格比普通款高多少？'].apply(extract_number)
    
    # 创建目标变量：是否会购买健康版产品
    # 基于健康版购买意愿（值>=3表示会购买）
    if '健康版购买意愿' in data.columns:
        data['会购买健康版'] = (data['健康版购买意愿'] >= 3).astype(int)
    elif '29.您是否会因为狗牙而推出健康版产品（如非油炸、全谷物）而增加购买意愿？' in df.columns:
        intention_map = {
            '一定会': 4,
            '可能会': 3,
            '不确定': 2,
            '可能不会': 1,
            '一定不会': 0
        }
        health_intention = df['29.您是否会因为狗牙而推出健康版产品（如非油炸、全谷物）而增加购买意愿？'].map(intention_map)
        data['会购买健康版'] = (health_intention >= 3).astype(int)
    
    # 标准化数值特征
    num_features = ['购买频率']
    if '健康关注度' in data.columns:
        num_features.append('健康关注度')
    if '健康版价格承受度' in data.columns:
        # 填充缺失值
        data['健康版价格承受度'] = data['健康版价格承受度'].fillna(data['健康版价格承受度'].mean())
        num_features.append('健康版价格承受度')
    
    # 填充缺失值并标准化
    for feature in num_features:
        if feature in data.columns:
            data[feature] = data[feature].fillna(data[feature].mean())
    
    # 只有在有足够的数值特征时才进行标准化
    if len(num_features) > 0:
        scaler = StandardScaler()
        data[num_features] = scaler.fit_transform(data[num_features])
    
    print(f"数据预处理完成。特征数量: {data.shape[1]-1 if '会购买健康版' in data.columns else data.shape[1]}, 样本数量: {data.shape[0]}")
    
    return data

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """训练XGBoost模型并评估"""
    print("正在训练XGBoost模型...")
    
    # 定义参数网格
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # 创建XGBoost分类器
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # 训练模型
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")
    
    # 在测试集上评估模型
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    classes = ['不会购买', '会购买']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵中添加文本标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix_improved.png')
    plt.show()
    
    return best_model

def plot_shap_analysis(model, X):
    """绘制SHAP分析图表"""
    print("正在生成SHAP分析图表...")
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # 创建输出目录
    os.makedirs('shap_plots', exist_ok=True)
    
    # 1. 摘要图 - 显示特征对模型输出的影响
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title('特征重要性排序（基于SHAP值）', fontsize=16)
    plt.tight_layout()
    plt.savefig('shap_plots/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 摘要点图 - 显示每个特征的SHAP值分布
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X, show=False)
    plt.title('特征SHAP值分布', fontsize=16)
    plt.tight_layout()
    plt.savefig('shap_plots/shap_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 获取前5个最重要的特征
    feature_importance = np.abs(shap_values).mean(0)
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    })
    top_features = feature_importance_df.sort_values('importance', ascending=False).head(5)['feature'].tolist()
    
    # 4. 为每个重要特征创建依赖图
    for feature in top_features:
        feature_idx = list(X.columns).index(feature)
        plt.figure(figsize=(10, 7))
        
        # 使用传统的依赖图方法，确保数据点显示
        try:
            shap.dependence_plot(
                feature_idx, 
                shap_values, 
                X, 
                show=False,
                alpha=0.8,  # 增加点的透明度，使重叠点更明显
                dot_size=100,  # 增大点的大小
                x_jitter=0.1  # 添加水平抖动以避免点重叠
            )
            plt.title(f'{feature}特征的SHAP依赖图', fontsize=16)
            plt.tight_layout()
            plt.savefig(f'shap_plots/dependence_{feature}.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"绘制{feature}的依赖图时出错: {e}")
        finally:
            plt.close()
    
    # 5. 添加一个热力图，显示特征之间的相关性
    try:
        plt.figure(figsize=(12, 10))
        # 只计算数值特征的相关性
        numeric_X = X.select_dtypes(include=['float64', 'int64'])
        if not numeric_X.empty:
            correlation = numeric_X.corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            cmap = plt.cm.RdBu_r
            
            sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                        square=True, linewidths=.5, annot=True, fmt='.2f', cbar_kws={"shrink": .5})
            
            plt.title('特征相关性热力图', fontsize=16)
            plt.tight_layout()
            plt.savefig('shap_plots/feature_correlation.png', dpi=300, bbox_inches='tight')
        else:
            print("没有数值特征可用于创建相关性热力图")
    except Exception as e:
        print(f"创建相关性热力图时出错: {e}")
    finally:
        plt.close()
    
    # 6. 添加力图，显示单个预测的解释
    # 选择一个正例和一个负例进行解释
    pos_indices = np.where(model.predict(X) == 1)[0]
    neg_indices = np.where(model.predict(X) == 0)[0]
    
    if len(pos_indices) > 0:
        try:
            pos_idx = pos_indices[0]
            plt.figure(figsize=(12, 8))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[pos_idx, :], 
                X.iloc[pos_idx, :],
                matplotlib=True,
                show=False
            )
            plt.title('正例预测解释（力图）', fontsize=16)
            plt.tight_layout()
            plt.savefig('shap_plots/force_plot_positive.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"创建正例力图时出错: {e}")
        finally:
            plt.close()
    
    if len(neg_indices) > 0:
        try:
            neg_idx = neg_indices[0]
            plt.figure(figsize=(12, 8))
            shap.force_plot(
                explainer.expected_value, 
                shap_values[neg_idx, :], 
                X.iloc[neg_idx, :],
                matplotlib=True,
                show=False
            )
            plt.title('负例预测解释（力图）', fontsize=16)
            plt.tight_layout()
            plt.savefig('shap_plots/force_plot_negative.png', dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"创建负例力图时出错: {e}")
        finally:
            plt.close()
    
    # 7. 添加特征重要性条形图（使用XGBoost的内置方法）
    try:
        plt.figure(figsize=(12, 8))
        xgb.plot_importance(model, max_num_features=15, height=0.6, importance_type='gain')
        plt.title('XGBoost特征重要性（基于增益）', fontsize=16)
        plt.tight_layout()
        plt.savefig('shap_plots/xgboost_importance.png', dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"创建XGBoost特征重要性图时出错: {e}")
    finally:
        plt.close()
    
    # 注意：移除需要graphviz的树可视化，因为用户环境中没有安装
    # if hasattr(model, 'get_booster'):
    #     plt.figure(figsize=(20, 12))
    #     xgb.plot_tree(model, num_trees=0, rankdir='LR')
    #     plt.title('XGBoost决策树（第一棵树）', fontsize=16)
    #     plt.tight_layout()
    #     plt.savefig('shap_plots/xgboost_tree.png', dpi=300, bbox_inches='tight')
    #     plt.close()
    
    print("SHAP分析图表已保存到 'shap_plots' 目录")

def main():
    """主函数"""
    # 设置文件路径
    excel_path = r"C:\Users\jiawang\Desktop\狗牙儿.xlsx"
    
    # 加载数据
    df = load_excel_data(excel_path)
    
    # 预处理数据
    data = preprocess_data(df)
    
    # 检查是否有足够的数据
    if len(data) < 10:
        print("数据量太少，无法进行有效的模型训练。")
        return
    
    # 检查是否有目标变量
    if '会购买健康版' not in data.columns:
        print("缺少目标变量'会购买健康版'，无法进行模型训练。")
        return
    
    # 分离特征和目标变量
    X = data.drop('会购买健康版', axis=1)
    y = data['会购买健康版']
    
    # 打印特征信息
    print(f"\n特征列表 ({len(X.columns)}个特征):")
    for i, col in enumerate(X.columns):
        print(f"{i+1}. {col}")
    
    # 检查数据平衡性
    class_counts = y.value_counts()
    print("\n目标变量分布:")
    for cls, count in class_counts.items():
        print(f"类别 {cls}: {count} 样本 ({count/len(y)*100:.2f}%)")
    
    # 处理可能存在的NaN值
    X = X.fillna(X.mean())
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 训练XGBoost模型
    model = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # 使用SHAP解释模型
    plot_shap_analysis(model, X)
    
    # 添加额外的可视化：特征分布
    print("\n生成特征分布图...")
    
    # 创建输出目录
    os.makedirs('feature_plots', exist_ok=True)
    
    # 为每个数值特征创建分布图
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    
    for feature in numeric_features:
        try:
            plt.figure(figsize=(10, 6))
            
            # 创建按目标变量分组的直方图
            sns.histplot(data=data, x=feature, hue='会购买健康版', kde=True, palette=['skyblue', 'salmon'])
            
            plt.title(f'{feature}的分布 (按购买意愿分组)', fontsize=14)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('频数', fontsize=12)
            plt.legend(['不会购买', '会购买'])
            plt.tight_layout()
            plt.savefig(f'feature_plots/{feature}_distribution.png', dpi=300)
        except Exception as e:
            print(f"创建{feature}分布图时出错: {e}")
        finally:
            plt.close()
    
    # 为分类特征创建计数图
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns
    
    for feature in categorical_features:
        try:
            plt.figure(figsize=(12, 8))
            
            # 创建按目标变量分组的计数图
            sns.countplot(data=data, x=feature, hue='会购买健康版', palette=['skyblue', 'salmon'])
            
            plt.title(f'{feature}的分布 (按购买意愿分组)', fontsize=14)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('频数', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.legend(['不会购买', '会购买'])
            plt.tight_layout()
            plt.savefig(f'feature_plots/{feature}_counts.png', dpi=300)
        except Exception as e:
            print(f"创建{feature}计数图时出错: {e}")
        finally:
            plt.close()
    
    # 创建二元特征的热力图
    binary_features = [col for col in X.columns if X[col].nunique() <= 2]
    
    if len(binary_features) > 1:
        try:
            plt.figure(figsize=(14, 12))
            
            # 计算二元特征之间的相关性
            binary_corr = X[binary_features].corr()
            
            # 创建热力图
            mask = np.triu(np.ones_like(binary_corr, dtype=bool))
            sns.heatmap(binary_corr, mask=mask, annot=True, cmap='coolwarm', fmt='.2f',
                        linewidths=0.5, cbar_kws={"shrink": .5})
            
            plt.title('二元特征之间的相关性', fontsize=16)
            plt.tight_layout()
            plt.savefig('feature_plots/binary_features_correlation.png', dpi=300)
        except Exception as e:
            print(f"创建二元特征热力图时出错: {e}")
        finally:
            plt.close()
    
    print("特征分布图已保存到 'feature_plots' 目录")
    
    print("\n分析完成！")

if __name__ == "__main__":
    main()
