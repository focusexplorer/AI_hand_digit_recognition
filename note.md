#实践感悟：
#神经网络的输入和输出值范围需要合理设计，权重矩阵的初始值需要随机选在合理范围内，以确保开始算出来的最终误差不太大，那么反馈调节权重矩阵时就不易振荡，且容易快速收敛。
#sigmoid容易饱合，故对于权重矩阵的选取一定要在合理范围内，如果落在饱和区，导数为0，反向回馈就传播不了。
#把所有计算化为矩阵形式来计算（总是可以找到一种数学形式），如果使用for循环来一一计算矩阵中每个元素，numpy的计算速度会大打折扣。
