# 可视化工具Tensorboard的简介安装与使用
## 一. 简介
再次回顾模型训练步骤

![1](pcs/1.png "1")

迭代训练可视化可以及时监控当前训练的训练状态，即使调控

![2](pcs/2.png "2")

## 二. 安装
pip install

## 三. 可视化使用
### 1. 基本接口，类，方法
#### （1）类：SummaryWriter
![3](pcs/3.png "3")

#### （2）方法
##### [1,2]. add_scalar(),add_scalars()
![4](pcs/4.png "4")

##### [3]. add_histogram()
![5](pcs/5.png "5")

##### [4]. 图像可视化：add_image 和 torchvision.utils.make_grid
![6](pcs/6.png "6")

![7](pcs/7.png "7")

##### [5]. 计算图可视化：add_graph()
![8](pcs/8.png "8")

##### [6]. 模型信息显示：torchsummary
![9](pcs/9.png "9")