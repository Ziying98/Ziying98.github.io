---
title: "心脏病预测分析与模型构建"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/heart-disease-prediction
date: 2024-01-15
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

### 2. 模型构建
对比两种经典分类模型：
```python
# 逻辑回归模型
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
```

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

### 模型性能对比
![ROC曲线对比](/images/portfolio/heart-disease-prediction/roc_curve.png)
- 随机森林模型AUC值达0.92，显著优于逻辑回归（0.89），具备更强的风险区分能力。

### 关键特征重要性
![特征重要性](/images/portfolio/heart-disease-prediction/feature_importance.png)
- 胸痛类型（cp）、最大心率（thalach）和血管数量（ca）是预测心脏病的Top3特征，临床意义明确。

---

This is an item in your portfolio. It can be have images or nice text. If you name the file .md, it will be parsed as markdown. If you name the file .html, it will be parsed as HTML. 
