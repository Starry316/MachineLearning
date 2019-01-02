# MachineLearning
Repository for Machine Learning Class  

模式分类的一些课后作业  
chapter2 实现了一个简单的pdf服从高斯分布的贝叶斯判别函数

chapter3 实现了最大似然参数估计

chapter4 实现了parzen窗和knn方法的非参数化估计

chapter6 实现了一个简单的全连接神经网络，可以进行任意层数的堆叠

一个适用于MNIST数据集的5层的神经网络构建方法示例如下（输入层为784，输出层为10）：
```
Network model = new Network()
# 向网络中添加层
model.addLayer(784, 28, actfun = "relu")
model.addLayer(28, 28, actfun = "relu")
model.addLayer(28, 28, actfun = "relu")
model.addLayer(28, 10, actfun = "relu")
# 训练
model.train(900000, train_images, y, val_data=test_data, val_label=test_labels, show_num=2000)
# 测试
model.test(test_data, test_labels)
```

chapter9 使用sklearn提供的svm和knn框架结合神经网络进行了集成学习实验。