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
# 2.强化学习框架
## *2.1 Gym/Gymnasium*
前置知识：Python、2.理论篇-强化学习  

主要内容：Basic Usage、Training an Agent、Create a Custom Environment、Recording Agents Speeding Up Training Compatibility with Gym、Migration Guide-v0.21 to v1.0.0、Env类、Make and register、Spaces类、Wrappers子库、Vectorize、Utility functions、Classic Control、Box2D、Toy Text	、MuJoCo、Atari、External Environments  

推荐资料：[Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)

## *2.2 算法开源框架*
前置知识：Python、2.理论篇-强化学习  

主要内容：baselines、stable-baselines3、spinningup、dopamine、rlpyt、PARL、CleanRL、ElegantRL、Deep Reinforcement Learning Algorithms with PyTorch、Tianshou、rainbow-is-all-you-need、PureJaxRL、OpenRL  

推荐资料：[强化学习相关框架整理（包含分布式多智能体）](https://blog.csdn.net/weixin_51775090/article/details/135745009)

## *2.3 多智能体/分布式开源框架*
前置知识：Python、2.理论篇-强化学习  

主要内容：Acme、rl_games、RLLib、Mava、Seed-rl  

推荐资料：[强化学习相关框架整理（包含分布式多智能体）](https://blog.csdn.net/weixin_51775090/article/details/135745009)

## *2.4 环境开源框架*
前置知识：Python、2.理论篇-强化学习  

主要内容：IsaacGym、EnvPool、Gymnax、Brax、Jumanji、OpenSpiel  

推荐资料：[强化学习相关框架整理（包含分布式多智能体）](https://blog.csdn.net/weixin_51775090/article/details/135745009)