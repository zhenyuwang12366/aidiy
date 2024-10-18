# 线性代数

本节将介绍线性代数中的基本数学对象、算术和运算，并用数学符号和相应的代码实现来表示它们。

## 标量

标量是最基本的数学对象，它只包含一个数值。例如，温度、质量或速度都是标量的例子。
(**标量由只有一个元素的张量表示**)。

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab paddle
import warnings
warnings.filterwarnings(action='ignore')
import paddle

x = paddle.to_tensor([3.0])
y = paddle.to_tensor([2.0])

x + y, x * y, x / y, x**y
```

## 向量

向量是标量的有序集合，每个标量称为向量的元素或分量。在数据科学中，向量常用于表示数据集中的样本。

例如,如果我们正在研究医院患者可能面临的心脏病发作风险，可能会用一个向量来表示每个患者，
其分量为最近的生命体征、胆固醇水平、每天运动时间等。

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

```{.python .input}
#@tab paddle
x = paddle.arange(4)
x
```

我们可以使用下标来引用向量的任一元素，例如通过$x_i$来引用第$i$个元素。

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$

其中$x_1,\ldots,x_n$是向量的元素。在代码中，我们(**通过张量的索引来访问任一元素**)。

```{.python .input}
x[3]
```

### 长度、维度和形状

向量只是一个数字数组，每个数组都有一个长度，每个向量也是如此。
一个向量$\mathbf{x}$由$n$个标量组成，可以将其表示为$\mathbf{x}\in\mathbb{R}^n$。
向量的长度通常称为向量的*维度*（dimension）。

与普通的Python数组一样，我们可以通过调用`len()`函数来[**访问张量的长度**]。

```{.python .input}
len(x)
```

当用张量表示一个向量（只有一个轴）时，我们也可以通过`.shape`属性访问向量的长度。
形状（shape）是一个元素组，列出了张量沿每个轴的长度（维数）。
对于(**只有一个轴的张量，形状只有一个元素。**)

```{.python .input}
x.shape
```

## 矩阵

矩阵是二维数组，可以看作是向量的推广。在编程中，矩阵通常表示为具有两个轴的张量。

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$


```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

```{.python .input}
#@tab paddle
A = paddle.reshape(paddle.arange(20), (5, 4))
A
```

我们可以通过索引来访问向量的特定元素。
如果没有给出矩阵$\mathbf{A}$的标量元素，我们可以简单地使用矩阵$\mathbf{A}$的小写字母索引下标$a_{ij}$来引用$[\mathbf{A}]_{ij}$。
也将逗号插入到单独的索引中，例如$a_{2,3j}$和$[\mathbf{A}]_{2i-1,3}$。

### 转置

矩阵的转置是通过交换其行和列得到的。在编程中，我们可以通过特定的函数来获取矩阵的转置。

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

现在在代码中访问(**矩阵的转置**)。

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

```{.python .input}
#@tab paddle
paddle.transpose(A, perm=[1, 0])
```

作为方阵的一种特殊类型，[***对称矩阵*（symmetric matrix）$\mathbf{A}$等于其转置：$\mathbf{A} = \mathbf{A}^\top$**]。
这里定义一个对称矩阵$\mathbf{B}$：

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab paddle
B = paddle.to_tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

现在我们将`B`与它的转置进行比较。

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

```{.python .input}
#@tab paddle
B == paddle.transpose(B, perm=[1, 0])
```


## 张量

张量提供了一种灵活的方式来表示多维数据结构，可以涵盖从一维到更高维度的各种数组。
比如，向量被视为一维张量，而矩阵则是二维张量。
这些数据结构的元素可以通过类似矩阵的索引方法访问，例如$x_{ijk}$或$[\mathsf{X}]_{1,2i-1,3}$。

特别是在图像处理领域，张量的应用非常广泛，因为图像本质上是由多个维度组成的数组，
这三个维度分别代表图像的高度、宽度以及一个*通道*（channel）维度，
这个通道维度用于存储颜色信息，如红色、绿色和蓝色通道。
```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

```{.python .input}
#@tab paddle
X = paddle.reshape(paddle.arange(24), (2, 3, 4))
X
```

## 张量算法的基本性质

[**给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量**]。
例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法。

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # 不能通过分配新内存将A克隆到B
A, A + B
```

```{.python .input}
#@tab paddle
A = paddle.reshape(paddle.arange(20, dtype=paddle.float32), (5, 4))
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
```

具体而言，[**两个矩阵的按元素乘法称为*Hadamard积*（Hadamard product）（数学符号$\odot$）**]。
对于矩阵$\mathbf{B} \in \mathbb{R}^{m \times n}$，
其中第$i$行和第$j$列的元素是$b_{ij}$。
矩阵$\mathbf{A}$和$\mathbf{B}$的Hadamard积为：
$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

```{.python .input}
#@tab paddle
A * B
```

将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
#@tab paddle
a = 2
X = paddle.reshape(paddle.arange(24), (2, 3, 4))
a + X, (a * X).shape
```

## 降维


我们可以对任意张量进行的一个有用的操作是[**计算其元素的和**]。
数学表示法使用$\sum$符号表示求和。
为了表示长度为$d$的向量中元素的总和，可以记为$\sum_{i=1}^dx_i$。
在代码中可以调用计算求和的函数：

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
#@tab paddle
x = paddle.arange(4, dtype=paddle.float32)
x, x.sum()
```

我们可以(**表示任意形状张量的元素和**)。
例如，矩阵$\mathbf{A}$中元素的和可以记为$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$。


```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

```{.python .input}
#@tab paddle
A.shape, A.sum()
```

默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。
我们还可以[**指定张量沿哪一个轴来通过求和降低维度**]。
以矩阵为例，为了通过求和所有行的元素来降维（轴0），可以在调用函数时指定`axis=0`。
由于输入矩阵沿0轴降维以生成输出向量，因此输入轴0的维数在输出形状中消失。


```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab paddle
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

指定`axis=1`将通过汇总所有列的元素降维（轴1）。因此，输入轴1的维数在输出形状中消失。

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab paddle
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和。


```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # 结果和A.sum()相同
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # 结果和tf.reduce_sum(A)相同
```

```{.python .input}
#@tab paddle
A.sum(axis=[0, 1])
```


计算平均值的函数可以沿指定轴降低张量的维度。

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

```{.python .input}
#@tab paddle
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

### 非降维求和


但是，有时在调用函数来[**计算总和或均值时保持轴数不变**]会很有用。

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab paddle
sum_A = paddle.sum(A, axis=1, keepdim=True)
sum_A
```

例如，由于`sum_A`在对每行进行求和后仍保持两个轴，我们可以(**通过广播将`A`除以`sum_A`**)。

```{.python .input}
A / sum_A
```

如果我们想沿[**某个轴计算`A`元素的累积总和**]，
比如`axis=0`（按行计算），可以调用`cumsum`函数。
此函数不会沿任何轴降低输入张量的维度。

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```


## 点积（Dot Product）

给定两个向量$\mathbf{x},\mathbf{y}\in\mathbb{R}^d$，
它们的*点积*（dot product）$\mathbf{x}^\top\mathbf{y}$
（或$\langle\mathbf{x},\mathbf{y}\rangle$）
是相同位置的按元素乘积的和：$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$。

[~~点积是相同位置的按元素乘积的和~~]

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
#@tab paddle
y = paddle.ones(shape=[4], dtype='float32')
x, y, paddle.dot(x, y)
```

注意，(**我们可以通过执行按元素乘法，然后进行求和来表示两个向量的点积**)：

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
#@tab paddle
paddle.sum(x * y)
```


## 矩阵-向量积

将矩阵$\mathbf{A}$用它的行向量表示：

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

其中每个$\mathbf{a}^\top_{i} \in \mathbb{R}^n$都是行向量，表示矩阵的第$i$行。
[**矩阵向量积$\mathbf{A}\mathbf{x}$是一个长度为$m$的列向量，
其第$i$个元素是点积$\mathbf{a}^\top_i \mathbf{x}$**]：

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

我们可以把一个矩阵$\mathbf{A} \in \mathbb{R}^{m \times n}$乘法看作一个从$\mathbb{R}^{n}$到$\mathbb{R}^{m}$向量的转换。
这些转换是非常有用的，例如可以用方阵的乘法来表示旋转。
以后我们也可以使用矩阵-向量积来描述在给定前一层的值时，
求解神经网络每一层所需的复杂计算。

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
# 使用`mv`函数, 会执行矩阵-向量积
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
# 使用与点积相同的`matvec`函数, 会执行矩阵-向量积
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
#@tab paddle
A.shape, x.shape, paddle.mv(A, x)
```

## 矩阵-矩阵乘法

假设有两个矩阵$\mathbf{A} \in \mathbb{R}^{n \times k}$和$\mathbf{B} \in \mathbb{R}^{k \times m}$：

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$



$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$
当我们简单地将每个元素$c_{ij}$计算为点积$\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$


在下面的代码中，我们在`A`和`B`上执行矩阵乘法。

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
#@tab paddle
B = paddle.ones(shape=[4, 3], dtype='float32')
paddle.mm(A, B)
```
