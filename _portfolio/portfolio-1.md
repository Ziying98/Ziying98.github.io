---
title: "心脏病预测分析与模型构建"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/heart-disease-prediction
date: 2026-01-17
excerpt: "基于心脏病数据集，通过数据分析与机器学习模型预测患病风险，对比逻辑回归与随机森林性能。"
header:
  teaser: /images/portfolio/heart-disease-prediction/roc_curve.png
tags:
  - 心脏病预测
  - 机器学习
  - 数据分析
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: Pandas
  - name: Matplotlib
  - name: Seaborn
---

## 项目背景
心脏病是全球致死率最高的疾病之一。本项目针对公开心脏病数据集进行全流程分析：从数据清洗预处理，到统计特征挖掘，再到构建机器学习模型预测患病风险，最终通过可视化手段呈现关键结论，为临床风险评估提供数据支持。


## 核心实现
### 1. 数据预处理
```python
# 删除重复值
df = df.drop_duplicates()
# 特征标准化（模型输入必备步骤）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2.分类模型构建
对比两种经典分类模型：
```python
# 逻辑回归模型
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
```
### 3. 聚类模型构建
使用 K 均值分析群体特征：
# 肘部法则确定最佳聚类数  
inertias = []  
silhouette_scores = []  
K_range = range(2,11)  
for k in K_range:  
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)  
    kmeans_temp.fit(X_train_scaled)  
    inertias.append(kmeans_temp.inertia_)  
    silhouette_scores.append(silhouette_score(X_train_scaled, kmeans_temp.labels_))  

# 构建K均值模型（k=2）  
kmeans_model = KMeans(n_clusters=2, random_state=42, n_init=10)  
kmeans_model.fit(X_train_scaled)  

# PCA降维可视化聚类结果  
pca = PCA(n_components=2)  
X_pca = pca.fit_transform(X_train_scaled)  

### 3. 模型评估
自定义评估函数输出关键指标：
```python
def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name} 评估结果:")
    print(f"准确率: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred))
```


## 分析结果
### 目标变量分布
![目标变量分布](/images/portfolio/heart-disease-prediction/target_distribution.png)
- 数据集患病样本占比54%，分布均衡，避免模型偏向性。

### 年龄分布特征
![年龄分布](/images/portfolio/heart-disease-prediction/age_distribution.png)
- 50-60岁是心脏病高发年龄段，需重点关注该群体的预防干预。

### 特征相关性
![相关性热力图](/images/portfolio/heart-disease-prediction/correlation_heatmap.png)
- 胸痛类型（cp）、最大心率（thalach）与患病风险高度相关，是核心预测因子。

### 分类模型性能对比
![ROC曲线对比](/images/portfolio/heart-disease-prediction/roc_curve.png)
- 随机森林模型AUC值达0.92，显著优于逻辑回归（0.89），具备更强的风险区分能力。

### 聚类模型分析
![K均值聚类肘部法则图](/images/portfolio/heart-disease-prediction/kmeans_elbow.png)
- 聚类数 k=2 时，轮廓系数最高且惯性下降趋缓，符合 “心脏病 / 无心脏病” 的二分类场景。

### 聚类结果可视化
![聚类结果可视化](/images/portfolio/heart-disease-prediction/kmeans_clustering.png)
- 左图：K 均值聚类将样本分为两类，群体特征区分明显；右图：真实标签分布与聚类结果高度吻合，说明聚类模型有效捕捉了患病群体的特征。

### 关键特征重要性
![特征重要性](/images/portfolio/heart-disease-prediction/feature_importance.png)
- 胸痛类型（cp）、最大心率（thalach）和血管数量（ca）是预测心脏病的Top3特征，临床意义明确。

---

This is an item in your portfolio. It can be have images or nice text. If you name the file .md, it will be parsed as markdown. If you name the file .html, it will be parsed as HTML. 
