[[数学基础]]
# MachineLearning

## 回归算法

### 多元线性回归

```python
from sklearn import datasets  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
  
# 加载糖尿病数据集  
diabetes = datasets.load_diabetes()  
X = diabetes.data  
y = diabetes.target  
  
# 将数据集划分为训练集和测试集  
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)  
  
# 创建一个多元线性回归算法对象  
lr = LinearRegression()  
  
# 使用训练集训练模型  
lr.fit(X_train, y_train)  
  
# 使用测试集进行预测  
y_pred_train = lr.predict(X_train)  
y_pred_test = lr.predict(X_test)  
  
# 打印模型的均方差  
print('均方误差：%.2f' % mean_squared_error(y_train, y_pred_train))  
print('均方误差：%.2f' % mean_squared_error(y_test, y_pred_test))
```

### 逻辑回归

二分类任务算法，适用于MultiLabel模型，将多分类问题转化为多个二分类，因为每个二分类之间是相互独立的

### SoftMax回归

多分类任务算法，适合MultiClass模型，因为互斥就是各类别概率之和必须为1

```python
from sklearn import datasets  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score  
  
iris = datasets.load_iris()  
X = iris.data  
y = iris.target  
# print(X)  
# print(y)  
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  
  
# 创建一个逻辑回归对象，这里逻辑回归会根据我们的数据决定是用二分类还是多分类  
  
lr = LogisticRegression(max_iter=1000)  
# lr = LogisticRegression(multi_class='ovr')  # 二分类  
# lr = LogisticRegression(multi_class='multinomial')  # Softmax回归  
  
lr.fit(X_train, y_train)  
  
y_pred = lr.predict(X_test)  
  
print('准确率: %.2f' % accuracy_score(y_test, y_pred))
```

### MultiClass与MultiLabel

- Multi-Class中每个Class之间是互斥的，如一个分类是猫就不能是狗和鸟，适用于图像分类等场景
- Multi-Label中每个实例可以有多个Label，如文本分类，一篇文章可以既属于科技分区又属于健康分区

## 损失函数

### 欠拟合与过拟合

![[截屏2024-08-22 14.21.33.png]]

### 正则化

正则化的目标是保持模型简单，以避免过拟合

$$Loss=BaseLoss+Penalty$$

- 回归任务

$$Loss=\underset{MSE}{\underline{MeanSquareError}}+Penalty$$

- 二分类任务

$$Loss=\underset{二分类交叉熵}{\underline{BinaryCrossEntropy}}+Penalty$$

- 多分类任务

$$Loss=\underset{多分类交叉熵}{\underline{MultiClassCrossEntropy}}+Penalty$$

##### L1、L2正则项

- L1正则项
$$Loss=BaseLoss+L1$$
L1 正则化是指在损失函数中加入 L1 范数，计算公式为 (lambda 为超参数)
$$\lambda\sum_{i=1}^{n}\left| w_i \right|$$
L1正则化的优点是可以减少特征数量，所以可以避免过拟合。缺点是可能会过度惩罚某些特征，使得一些有用的特征被舍弃。


- L2正则项
$$Loss=BaseLoss+L2$$
L2 正则项会将所有系数的权重都变得较小，但不为 0。

$$\lambda\sum_{i=1}^{n} w_i^2 $$
L1正则化的优点是可以降低所有特征的权重，防止过拟合。缺点是不会减少特征的数量，因此不能用于特征选择。

如果是多元线性回归的损失函数改为如下公式，叫做 Lasso 回归：

$$Loss=MSE+\lambda\sum_{i=1}^n\left|w_i\right|$$
如果是多元线性回归的损失函数改为如下公式，叫做Ridge岭回归：

$$Loss=MSE+\lambda\sum_{i=1}^nw_i^2$$

### 梯度下降法 #重要 

梯度下降法是一种非常重要的优化算法，能帮助我们在大规模的数据集上求解复杂的机器学习模型，通过不断的调参来逼近**最优解**模型。

![[截屏2024-08-22 20.00.32.png]]
1. 初始化模型参数，例如随机初始化权重
   $$\theta=w_0,w_1,w_2,...,w_n$$
2. 根据当前的模型参数计算出损失函数J(θ)的梯度
   $$gradient=\frac{\partial J(\theta)}{\partial \theta}$$
3. 根据损失函数的梯度更新模型参数
   $$\theta=\theta-\alpha\frac{\partial J(\theta)}{\partial \theta}$$
   其中α是学习率，用来控制每次迭代的步长。如果梯度为正，要减小θ；反之亦然。
4. 重复 2 与 3直到满足终止条件

### 小批量梯度下降

当数据量大的时候一次性读取训练集所有数据进行一次梯度下降迭代，很可能因为算力不足无法运行。
![[截屏2024-08-23 14.20.39.png]]
可以像上图一样将训练集数据分为多个**批次**（batch），每个批次所包含的数据量（samples）叫做**批次大小**（batch_size），梯度下降每次迭代使用一个批次的数据。
$$\theta=\theta-\alpha\frac{1}{b}\sum_{i=1}^b\frac{\partial J_i(\theta)}{\partial \theta}$$
##### 不同的梯度下降法

![[截屏2024-08-23 14.33.41.png]]
圈是 Loss 等高线，相当于一个山谷，+号位置为 Loss 最小处


![[截屏2024-08-23 14.42.50.png]]
训练数据交给机器从头到尾学习一遍是不够的，需要学习很多遍，每学习一遍我们称为学习一个**轮次**（epoch），所以小批量梯度下降就是`分轮次分批次的训练`。

每一批次的步骤同[[#梯度下降法 重要]]，只不过 2、3 步是一个 batch 的数据进行计算。