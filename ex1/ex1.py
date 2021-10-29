import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# 显示前五行数据
print(data.head())

# 数据描述
# print(data.describe())
# mean：平均值 std：标准差

# 可视化 scatter：散点
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8))


# print(plt.show())

# 代价函数
def CostFunction(X, y, theta):
    inner = np.power(((X @ theta.T) - y), 2)
    return sum(inner) / (2 * len(X))


# 在训练集最前面添加一列值全为1，方便进行向量计算
data.insert(0, 'Ones', 1)
print(data.head())


# 矩阵的列数 data.shape[0]为行数
cols = data.shape[1]
# X是所有行去掉最后一列的数据
X = data.iloc[:, 0:cols - 1]
# y是最后一列数据
y = data.iloc[:, cols - 1:cols]
# .iloc用法：[x1:x2,y1:y2]，逗号前是行逗号后是列

# print(X.head())
# print(y.head())

# 将X，y转换为矩阵
X = np.array(X.values)
y = np.array(y.values)
# 计算代价函数，theta初始值为0
theta = np.array([0, 0]).reshape(1, 2)

print(theta.shape)
print(X.shape)
print(y.shape)

# 结果是2倍？
print(CostFunction(X, y, theta))

'''梯度下降'''


# alpha:学习率 iters：迭代次数
def GradientDescent(X, y, theta, alpha, iters):
    # np.zeros:返回一个给定形状和类型的用0填充的数组
    # 初始化一个存放theta的临时矩阵(1, 2)
    temp = np.array(np.zeros(theta.shape))
    # 记录每次迭代计算的代价值
    cost = np.zeros(iters)
    m = X.shape[0]  # 样本数量

    for i in range(iters):
        # 向量化一步求解
        # 要用np.dot进行矩阵相乘
        temp = theta - (alpha / m) * ((X @ theta.T) - y).T @ X
        theta = temp
        cost[i] = CostFunction(X, y, theta)

    return theta, cost


alpha = 0.01
iters = 2000

final_theta, cost = GradientDescent(X, y, theta, alpha, iters)
print(final_theta)
# print(CostFunction(X, y, final_theta))

final_cost = CostFunction(X, y, final_theta)
print(final_cost)

population = np.linspace(data.Population.min(), data.Population.max(), 97)
profit = final_theta[0, 0] + (final_theta[0, 1] * population)

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(population, profit, 'r', label='Prediction')
ax.scatter(data['Population'], data['Profit'], label='Traning Data')
ax.legend(loc=4)  # 4表示在右下角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Prediction Profit by. Population Size')
print(plt.show())

fig,ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Traning Iters')
print(plt.show())