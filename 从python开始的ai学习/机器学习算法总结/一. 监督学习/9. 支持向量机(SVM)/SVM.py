# 导入所需的库和数据集
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# 加载数据集 & CSV文件路径
path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(path, 'boston_housing.csv')
# 从CSV文件中读取数据 
boston_df = pd.read_csv(file_path)
# 分离特征和目标变量 
X = boston_df.drop('TARGET', axis=1)
y = boston_df['TARGET']

# 数据预处理（标准化）
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建SVM模型并进行训练
svm = SVR(kernel='linear')
svm.fit(X_train, y_train)

# 对测试集进行预测
y_pred = svm.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print("均方误差（MSE）：", mse)

# 可视化预测结果
plt.scatter(range(len(y_test)), y_test, color='b', label='Actual')
plt.plot(range(len(y_test)), y_pred, color='r', label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Price')
plt.title('SVM Regression')
plt.legend()
plt.show()