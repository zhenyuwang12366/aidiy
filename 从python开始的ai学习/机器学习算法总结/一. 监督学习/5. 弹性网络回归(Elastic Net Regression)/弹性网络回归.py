import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
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

# 分隔数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化ElasticNet回归模型
elastic_net = ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42)

# 拟合模型
elastic_net.fit(X_train_scaled, y_train)

# 在测试集上预测
y_pred = elastic_net.predict(X_test_scaled)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print('Root Mean Squared Error: ', rmse)