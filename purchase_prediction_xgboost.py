import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import shap

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_and_preprocess_data(file_path):
    """加载并预处理数据"""
    print("正在加载和预处理数据...")
    
    # 加载数据
    df = pd.read_csv(file_path)
    
    # 重命名列以便更容易理解
    df.columns = ['购买动机', '口味偏好', '健康版价格承受度', '普通版价格承受度', '购买频率']
    
    # 处理多标签特征
    mlb_motives = MultiLabelBinarizer()
    mlb_flavors = MultiLabelBinarizer()
    
    # 将字符串列表转换为实际的Python列表
    df['购买动机'] = df['购买动机'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df['口味偏好'] = df['口味偏好'].apply(lambda x: eval(x) if isinstance(x, str) else [])
    
    # 转换多标签特征
    motives_encoded = pd.DataFrame(
        mlb_motives.fit_transform(df['购买动机']),
        columns=[f'动机_{c}' for c in mlb_motives.classes_]
    )
    
    flavors_encoded = pd.DataFrame(
        mlb_flavors.fit_transform(df['口味偏好']),
        columns=[f'口味_{c}' for c in mlb_flavors.classes_]
    )
    
    # 处理购买频率
    frequency_map = {
        '每天': 5,
        '每周数次': 4,
        '每周一次': 3,
        '每月数次': 2,
        '偶尔': 1
    }
    df['购买频率_数值'] = df['购买频率'].map(frequency_map)
    
    # 创建目标变量：是否会购买健康版产品
    # 如果健康版价格承受度有值，则认为会购买
    df['会购买健康版'] = df['健康版价格承受度'].notna().astype(int)
    
    # 合并所有特征
    features = pd.concat([
        motives_encoded, 
        flavors_encoded,
        df[['购买频率_数值']]
    ], axis=1)
    
    # 添加普通版价格承受度作为特征
    # 但不包括健康版价格承受度，避免数据泄露
    features['普通版价格承受度'] = df['普通版价格承受度'].fillna(df['普通版价格承受度'].mean())
    
    # 标准化数值特征
    scaler = StandardScaler()
    num_features = ['购买频率_数值', '普通版价格承受度']
    features[num_features] = scaler.fit_transform(features[num_features])
    
    # 目标变量
    target = df['会购买健康版']
    
    print(f"数据预处理完成。特征数量: {features.shape[1]}, 样本数量: {features.shape[0]}")
    
    return features, target, mlb_motives, mlb_flavors

def train_xgboost_model(X_train, y_train, X_test, y_test):
    """训练XGBoost模型并评估"""
    print("正在训练XGBoost模型...")
    
    # 设置参数
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # 创建并训练模型
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"模型准确率: {accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['不购买健康版', '购买健康版']))
    
    # 混淆矩阵
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    
    # 使用matplotlib替代seaborn的热力图
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['不购买健康版', '购买健康版'], rotation=45)
    plt.yticks(tick_marks, ['不购买健康版', '购买健康版'])
    
    # 在每个单元格中添加数字
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return model

def shap_analysis(model, X_test, feature_names):
    """使用SHAP进行模型可解释性分析"""
    print("正在进行SHAP可解释性分析...")
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 汇总图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('特征对购买健康版产品决策的影响（SHAP值）')
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.show()
    
    # 特征重要性图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.title('特征重要性（基于SHAP值）')
    plt.tight_layout()
    plt.savefig('shap_importance.png')
    plt.show()
    
    # 依赖图（针对最重要的特征）
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    top_features = shap_df.abs().mean().sort_values(ascending=False).head(3).index
    
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X_test, feature_names=feature_names, show=False)
        plt.title(f'{feature}对购买决策的影响')
        plt.tight_layout()
        plt.savefig(f'shap_dependence_{feature.replace(" ", "_")}.png')
        plt.show()

def main():
    """主函数"""
    # 加载和预处理数据
    X, y, mlb_motives, mlb_flavors = load_and_preprocess_data('consumer_data.csv')
    
    # 特征名称列表
    feature_names = list(X.columns)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
    
    # 训练模型
    model = train_xgboost_model(X_train, y_train, X_test, y_test)
    
    # 特征重要性
    plt.figure(figsize=(12, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.6)
    plt.title('XGBoost特征重要性')
    plt.tight_layout()
    plt.savefig('xgboost_importance.png')
    plt.show()
    
    # SHAP分析
    shap_analysis(model, X_test, feature_names)
    
    # 输出购买健康版产品的关键因素
    print("\n购买健康版产品的关键因素分析完成！")
    print("请查看生成的图表以了解影响消费者购买决策的主要因素。")

if __name__ == "__main__":
    main()
