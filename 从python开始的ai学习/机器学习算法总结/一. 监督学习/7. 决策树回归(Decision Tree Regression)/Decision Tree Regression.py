# 导入所需的库和模块
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import export_graphviz
import pydot
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

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树回归模型
regressor = DecisionTreeRegressor(max_depth=4)

# 训练模型
regressor.fit(X_train, y_train)

# 预测
y_pred = regressor.predict(X_test)

# 计算平均绝对误差和均方根误差
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)

# 可视化决策树模型
out_file = os.path.dirname(__file__)
dot_data = export_graphviz(regressor, out_file=None, feature_names=X.columns, filled=True)
graph = pydot.graph_from_dot_data(dot_data)[0]
graph.write_png(os.path.join(out_file, 'decision_tree.png'))