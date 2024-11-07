# transforms图像增强（一）
## 一. 什么是数据增强
数据增强又称为数据增广，数据扩增，它是对**训练集**进行变换，使训练集更丰富，从而让模型更具**泛化能力**

类比

![1](pcs/1.png "1")

例子：

![2](pcs/2.png "2")

## 二. transforms————裁剪
### 1. transforms.CenterCrop

![3](pcs/3.png "3")

### 2. transforms.RandomCrop

![4](pcs/4.png "4")

![5](pcs/5.png "5")

### 3. transforms.RandomResizedCrop

![6](pcs/6.png "6")

### 4. transforms.Five(Ten)Crop

![7](pcs/7.png "7")

## 三. transforms————翻转，旋转
### 1. RandomHorizontalFlip 和 RandomVerticalFlip

![8](pcs/8.png "8")

### 2. RandomRotation

![9](pcs/9.png "9")

