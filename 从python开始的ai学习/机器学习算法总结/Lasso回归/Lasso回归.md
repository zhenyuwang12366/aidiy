# Lasso回归
## 案例介绍
在这个案例中，我们将使用波士顿房屋数据集来演示Lasso回归的应用。波士顿房屋数据集包含了对波士顿不同地区房屋价格的预测因素，如犯罪率、房产税率等，以及对应地区的房屋价格中位数。

我们的目标是使用Lasso回归算法来建立一个预测模型，通过输入特征预测房屋价格中位数。
## 算法原理
Lasso回归是一种线性回归的扩展方法，它通过加入L1正则化项来进行特征选择和模型参数的压缩。Lasso回归的优化目标函数如下：
```math
\mathrm{minimize}\bigg(\sum\limits_{i=1}^{n}{(y_i-\hat{y_i})^2}+\alpha\sum\limits_{j=1}^{p}{\lvert\beta_j\rvert}\bigg)
```
其中, $n$ 是样本数量, $p$ 是特征数量, $y_i$ 是实际观测值, $\hat{y_i}$ 是预测值, $\beta_j$ 是模型的参数（系数）, $\alpha$ 是L1正则化项的权重。

L1正则化项通过在优化过程中将某些模型参数变为0来实现特征选择的作用。较大的 $\alpha$ 值将更多的系数压缩到零，从而选择出更少的特征。较小的 $\alpha$ 值将保留更多的特征。
## 数据集
我们将使用波士顿房屋数据集。这个数据集包含 506 个样本和 13 个特征。每个样本都有一个房屋价格中位数作为标签。
## 计算步骤
1. 导入数据集并进行数据预处理。
2. 拆分数据集为训练集和测试集。
3. 初始化Lasso回归模型，并训练模型。
4. 评估模型在测试集上的性能。
5. 可选：调整超参数来改善模型性能。
