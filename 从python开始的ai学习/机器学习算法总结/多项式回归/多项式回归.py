import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


pwd = os.path.dirname(os.path.realpath(__file__))
path = os.makedirs(os.path.join(pwd,'pcs'), exist_ok=True)
# 导入加州房屋数据
california = fetch_california_housing()

# 提取特征,目标变量,和特征名
X = california.data
y = california.target
X_name = california.feature_names
print(X_name)
# 将特征向量转换为多项式特征， 即生成公式推导中的矩阵形式, 这里是二次多项式
poly = PolynomialFeatures(degree = 2)  # 实例化poly
X_poly = poly.fit_transform(X)

# 使用多项式回归模型进行拟合，将多项式回归转换成线性回归，从矩阵的形式来看，其实都是 Ax = b
model = LinearRegression() 
model.fit(X_poly, y)

# 预测新的房屋价格
feature_num = 3
X_new = X[feature_num].reshape(1, -1)
X_new_poly = poly.transform(X_new)
y_new = model.predict(X_new_poly)

# 计算模型的性能指标
y_pred = model.predict(X_poly)
mse = mean_squared_error(y, y_pred)

print(mse)
# 绘制原始数据散点图和拟合曲线图
plt.scatter(X[:, feature_num], y, color = 'blue', label = 'Actual', alpha=0.3)
plt.scatter(X_new[:, feature_num], y_new, color = 'red', label = 'Prediction')
plt.plot(X[:, feature_num], y_pred, color = 'green', label = 'Regression', alpha = 0.5)
plt.xlabel('Population')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'{pwd}/pcs/image.png')