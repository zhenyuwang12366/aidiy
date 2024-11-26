import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 加载数据集 & CSV文件路径
path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(path, 'boston_housing.csv')
# 从CSV文件中读取数据 
boston_df = pd.read_csv(file_path)
# 分离特征和目标变量 
X = boston_df.drop('TARGET', axis=1)
y = boston_df['TARGET']

# 2. 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 初始化随机森林回归模型
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# 4. 在训练集上拟合模型
rf.fit(X_train, y_train)

# 5. 在测试集上进行预测
y_pred = rf.predict(X_test)

# 6. 评估模型性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# 显示特征的重要性
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.barh(range(X.shape[1]), feature_importances[indices], align='center')
plt.yticks(range(X.shape[1]), X.columns)
plt.xlabel("Relative Importance")
plt.show()