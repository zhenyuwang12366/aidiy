# 1.深度学习框架
主流的深度学习框架为PyTorch、Keras、TensorFlow，需至少掌握一个

## 1.1 PyTorch
前置知识：NumPy  

主要内容：张量简介与创建、张量操作与线性回归、计算图与动态图机制、autograd与逻辑回归、Dataloader与Dataset、transforms模块机制、transforms图像增强与多种数据预处理方法、模型创建步骤与nn.Module、卷积层、池化-线性-激活函数层、权值初始化、损失函数、优化器optimizer、学习率调整策略、Tensorboard的简介安装与使用、hook函数与CAM可视化、weight_decay、Dropout、Batch Normalization、更多Normalization方法、模型finetune、GPU的使用、Pytorch常见报错  

推荐资料：[从python开始的ai学习-深度学习 pytorch](https://github.com/Discrete-Mathematics/ai-self-learning/tree/main/%E4%BB%8Epython%E5%BC%80%E5%A7%8B%E7%9A%84ai%E5%AD%A6%E4%B9%A0/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%20pytorch)

## 1.2 Keras
前置知识：NumPy、*sklearn*  

主要内容：顺序模型、函数式API、Keras模型、Sequential顺序模型、Model类（函数式API）、Keras网络层、核心网络层、卷积层、池化层、局部连接层、循环层、嵌入层、融合层、高级激活层、标准化层、噪声层、层封装器、编写你自己的层、图像预处理、损失函数、评估标准、优化器、激活函数、回调、常用数据集、应用模块、后端、初始化、正则化、约束项、可视化、Scikit-learn API、工具、手写数字识别、CIFAR-10 CNN、CIFAR-10 ResNet、卷积滤波器可视化、卷积LSTM、Deep Dream、图片OCR、GAN辅助分类器

推荐资料：[Keras中文文档](https://keras-zh.readthedocs.io/)

## 1.3 TensorFlow
前置知识：*Keras*

主要内容：Eager Execution、张量、变量、自动微分、图、函数、模块、层、模型、训练循环、高级自动微分、不规则张量、洗漱张量、Numpy API、Tensor切片、*Sequential模型、Functional API、使用内置方法进行训练和评估、通过子类化构建新层和模型、保存并加载Keras模型、使用预处理层、自动编写回调、迁移学习和微调、使用TensorFlow Cloud训练Keras模型*、创建操作、生成随机数字、tf。data、优化流水线性能、分析流水线性能、检查点、SaveModel、使用GPU运行、*使用TPU运行*、使用tf.function提升性能、分析TensorFlow的性能、优化GPU性能、图优化、混合精度、Estimator  

推荐资料：[TensorFlow Core](https://tensorflow.google.cn/guide?hl=zh-cn)

---
# *2.图像处理库*

## *2.1 OpenCV*
前置知识：NumPy、计算机图形学

主要内容：读入图片、imshow()、保存图片、读取视频、截取图片、按颜色通道提取图片、按颜色通道合并图片、对图片边界填充、图像融合、更改图片长宽比、转换图片的颜色空间、图像阈值、图像平滑、腐蚀操作、膨胀操作、开运算、闭运算、梯度运算、sobel算子、Scharr算子、laplacian算子、Canny边缘检测、图像金字塔、拉普拉斯金字塔、图像轮廓、轮廓特征、轮廓近似、边界矩形、直方图、直方图均衡化、模板匹配、傅里叶变换  

推荐资料：[OpenCV最详细入门（一）-python（代码全部可以直接运行）](https://blog.csdn.net/WUHU648/article/details/118491096)  
　　　　　[OpenCV最详细入门（二）-python（代码全部可以直接运行）](https://blog.csdn.net/WUHU648/article/details/118580542)

## *2.2 Torchvision*
前置知识：*PyTorch*  

主要内容：models包、transforms包、datasets包、utils包  

推荐资料：[PyTorch：Torchvision的简单介绍与使用](https://blog.csdn.net/baidu_38797690/article/details/122513894)  

## *2.3 Scikit-Image*
前置知识：Numpy、计算机图形学  

主要内容：io子模块、color子模块、data子模块、filters子模块、draw子模块、transform子模块exposure子模块、feature子模块、graph子模块、measure子模块、morphology子模块、novice子模块、restoration子模块、segmentation子模块、util子模块、viewer子模块

## *2.4 Pillow*
前置知识：Numpy、计算机图形学  

主要内容：使用Image.open()创建图像示例、读写图像、剪贴图像、粘贴图像、合并图像、几何变换、颜色变换、图像增强、高级增强、动态图像、Postscript打印、配置加载器draft、构建图像、图像处理、Image对象方法、ImageDraw模块

推荐资料：[Python 图像处理 Pillow 库 基础篇](https://zhuanlan.zhihu.com/p/58671158)  
　　　　　[Python Pillow 库 Image 模块 构建图像 、 图像处理 与 Image对象方法](https://zhuanlan.zhihu.com/p/58926599)  
　　　　　[Python Pillow 库 ImageDraw 绘制图像模块](https://zhuanlan.zhihu.com/p/59849190)
