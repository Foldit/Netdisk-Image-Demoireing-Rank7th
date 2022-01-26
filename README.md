# 赛题介绍

选手需要建立模型，对比赛给定的带有摩尔纹的图片进行处理，消除屏摄产生的摩尔纹噪声，还原图片原本的样子，并提交模型输出的结果图片。

# 数据集简介

本次比赛的数据集所有的图像数据均由真实场景采集得到，再通过技术手段进行相应处理，生成可用的脱敏数据集。该任务为image-to-image的形式，因此源数据和GT数据均以图片的形式来提供。本次比赛不限制使用额外的训练数据来优化模型。测试数据集的GT不做公开。

数据集构成
```
|- root  
    |- images
    |- gts
```
本次比赛最新发布的数据集共包含训练集、A榜测试集、B榜测试集三个部分，其中训练集共1000个样本，A榜测试集共200个样本，B榜测试集共200个样本；
images 为带摩尔纹的源图像数据，gts 为无摩尔纹的真值数据（仅有训练集数据提供gts ，A榜测试集、B榜测试集数据均不提供gts）；
images 与 gts 中的图片根据图片名称一一对应。

以下图片为数据中的样本，左侧图片为原图，右侧图片为已经去除摩尔纹，同时亮度也有所调整的GT图(GroundTruth)。

<div>
    <img src="https://ai-studio-static-online.cdn.bcebos.com/6f9bc21753cc4c8c9f8f525c1b548380841d01d2eefa478f93d676522f67219f" width=300/>
        <img src="https://ai-studio-static-online.cdn.bcebos.com/2e9c2bf7c1e846ba82c2d1026325efdd592496baea49486088d6b4e8009e1db9" width=300/>
    
</div>

<div>
    <img src="https://ai-studio-static-online.cdn.bcebos.com/a15d81c472904748b722f20fa5e7338694df70951c374caab36c91d8eb92ce5e" width=300/>
        <img src="https://ai-studio-static-online.cdn.bcebos.com/0d0caba1312c4feda1e80c24b90b81597caf6fee12044f909c6d0ddd12ea0a3e" width=300/>
    
</div>




# Baseline模型WDNet介绍

## 整体结构
WDNet是ECCV 2020提出一种去除摩尔纹的模型。该模型是一种基于小波与双分支的神经网络，结构如下：

![](https://ai-studio-static-online.cdn.bcebos.com/1352cb0a68d14622a2b2e5d2ec3f3edb82deea6547b54eaf8aa5bb0a9e22ed24)

首先RGB图片需要通过WaveletTransform模块进行转换，得到一个48通道的数据，通过WDNet网络同样得到一个通道数与尺寸不变的特征图。最后在一次通过WaveletTransform使用转置卷积将图片还原得到最终预测结果。

这里WaveletTransform的权重是固定不变不需要训练的。

## DenseNet

![](https://ai-studio-static-online.cdn.bcebos.com/9ef3defdba2f4b15ba6bdde87cda1a724530b8cb3ade4593ad9637593f58b39c)

DenseNet中使用旁路连接和特征复用的方式缓解了梯度消失的问题，同时减少了网络参数。DenseNet已经被用于去雾和超分辨率网络。

如上图所示，该模型中的dense分支新增了一个方向感知模块（DPM），用于找到摩尔纹的方向。DPM的输出和每一个dense的输出相乘，然后乘以一个因子β然后与输入相加。该设计可以有效的定位摩尔纹的位置。


## Dilation
![](https://ai-studio-static-online.cdn.bcebos.com/8545d61038ad41658c5610ed14b9bcfa6ae9c1947cb9469c80fc2ccfb62b9b55)

下采样和池化可以增大感受野，但同时也丢失了一些细节。空洞卷积可以解决这个问题。在每一个dilation分支里，都有两层，有一个3x3的空洞卷积和3x3的普通卷积组成。
## 思路步骤
基于WDNet模型实现的baseline，对摩尔纹图像进行观察，增加了随机度数旋转和随机裁剪的数据增强策略和对loss权值的修改，以及修改学习率衰减的策略。

## 代码组织结构
```
demoire-baseline/
├── train.py
    └── dataset.py
    └── transforms.py
    └── vgg.py
    └── losses.py
    └── model.py
    
├── predict.py
    └── utils.py
        └── train_result/model/epoch_1200/model.pdparams
    └── model.py
    
```

## 数据增强策略
在RandomHorizontalFlip、Resize、Normalize的基础上,

1)因摩尔纹是以一定曲线的状态存在，所以对图像进行0-90度的随机旋转(Rotate);

2)直接将图像resize成一个较小图片，可能会损失图像上摩尔纹的信息，所以先裁取大小为(512,512)的图像，再resize成(512,512)大小(Crop)。

其中，根据https://aistudio.baidu.com/paddle/forum/topic/show/993042 可知，尺寸不变的resize不会与原图像不同。
![](https://ai.bdstatic.com/file/8FEB9634A75F4BC881CF7DFDCFD39815)
## 调参优化策略
1）使用baseline的原loss函数形式，preceptual loss的权值的改变对结果有增益，将preceptual loss的权值由1改为1.1。

2）学习率衰减采用余弦退火(CosineAnnealingDecay),在训练时梯度下降算法可能陷入局部最小值，此时可以通过突然提高学习率，来“跳出”局部最小值并找到通向全局最小值的路径。

3）较小的batch_size和较多的epoch对结果有增益,其中batch_size = 4，epoch = 1200。

4）优化器选择AdamW。
# 训练与预测
```
运行main.ipynb
```
## 在线训练与预测链接
https://aistudio.baidu.com/aistudio/projectdetail/3439230?contributionType=1
# 参考baseline链接
https://aistudio.baidu.com/aistudio/projectdetail/3220041?channelType=0&channel=0
