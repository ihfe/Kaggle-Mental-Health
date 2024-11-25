# Kaggle-Mental-Health

记录自己参加Kaggle比赛的项目，
都是很基础的操作，
最最关键的就是**数据处理+模型选择**；

我这里选择了逻辑回归、Adaboost和Xgboost做这个任务

上传`submission.csv`之后得到的`Public Score`如下表所示（可以理解为准确率）

|              | 逻辑回归 | XGBoost | Adaboost |
| ------------ | -------- | ------- | -------- |
| `Test`正确率 | 0.93965  | 0.94243 | 0.93848  |

### 一、数据处理

数据处理部分没什么好说的，做多了见的多了就会了

### 二、模型选择

##### 思考❓为什么可以选择这个XGboost模型做二分类？

- 首先XGboost可以解决分类问题，也可以解决回归问题，原理就是新一轮产生的模型=上一轮模型+决策树；而决策树是通过拟合上一轮模型预测结果的残差产生的。
- 放到这个问题中，这个问题是一个二分类问题，即结果要么0，要么1。XGboost的预测结果是概率形式（因为XGboost的损失函数是对数损失函数，结果为0.7就代表为1，结果为0.2就代表为0），这样模型预测的时候就会有残差，然后我们就可以使用新的弱学习器去拟合这个残差让我们整体的模型变得更加优秀。

##### 思考❓为什么可以选择这个Adaboost模型？

* Adaboost本来就是主要用于解决分类问题。（为什么主要用于解决分类问题？）

* Adaboost中每个弱学习器单独训练，第一个弱学习器预测出分类结果之后，我们需要增大误分类样本的权重，然后后面的弱学习器去拟合更改过权重的样本。

* 主要思想：`Boosting`

**学习🌳 怎么画出cm图**:

【固定代码，用的时候直接复制粘贴】

```python
sns.heatmap(cm_xgb,cmap='Blues',cbar=True,annot=True,fmt="d")
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
```

注意的点❗二者参数设定不一样的地方：

【但是`xgboost`和`lightgbm`参数设定的方式几乎一样】

```python
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False,
}

# Train XGBoost
xgb_model_s = xgb.XGBClassifier(**xgb_params)
xgb_model_s.fit(x_train1, y_train1)
y_pred_xgb_s = xgb_model_s.predict(x_test1)
```

但是`adaboost`没有`objective`等参数，在我的项目里，`adaboost`是这样设置的：

```python
base_estimator = DecisionTreeClassifier(max_depth=2)
ada_clf_s = AdaBoostClassifier(estimator=base_estimator,
    n_estimators=300,  # 300 个弱分类器
    learning_rate=0.1, # 学习率
    random_state=42,
)
ada_clf_w = AdaBoostClassifier(estimator=base_estimator,
    n_estimators=300,  # 300 个弱分类器
    learning_rate=0.1, # 学习率
    random_state=42,
)
```

