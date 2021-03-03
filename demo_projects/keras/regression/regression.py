#回归问题
from keras.datasets import boston_housing
from keras import layers
from keras import models
import numpy as np
from utility.timer import start_time,end_time
from utility.graphic import epochs_mae
from keras import regularizers
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
#print(train_data.shape)
#数据标准化(假设数据服从正态分布)，将取值范围差距很大的数据分布转化为标准分布,即将非标准正态分布转化为(0,1)的标准正态分布
mean = train_data.mean(axis=0)
train_data -= mean
#x - u的标准差和x的标准差是一样的，所以std用减去均值后的train_data来算也是可以的
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /=std

def build_model():
    """
    在具体实现dropout的时候可以在训练的时候手动dropout一些位置，然后再以相同倍率扩大输出
       这样即做到了在训练的时候去掉一些特征值避免过拟合，又使得与测试的时候该层的输出值大小相当，因为
       测试的时候没有舍弃特征，有更多的单元被激活，比如如下实现
       layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
        layer_output /= 0.5
    """
    model = models.Sequential()
    model.add(layers.Dense(54,activation='relu',input_shape=(train_data.shape[1],),                      kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

#K折验证。使用K折验证的原因是如果验证集比较小，那么验证集可能不会太好，0

k = 4
#/返回的是浮点数,//返回的是不大于商的整数
num_val_samples = len(train_data) // k
num_epochs = 80
all_scores = []
start = start_time()
# for i in range(k):
#     val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
#     partial_train_data = np.concatenate((train_data[:i * num_val_samples],
#                                          train_data[(i + 1) * num_val_samples:]), axis=0)
#     partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
#                                          train_targets[(i + 1) * num_val_samples:]], axis=0)
#     """
#     fit:
#     verbose：日志显示
#
#     verbose = 0 为不在标准输出流输出日志信息
#
#     verbose = 1 为输出进度条记录
#
#     verbose = 2 为每个epoch输出一行记录
#     evaluate:
#     verbose：日志显示
#
#     verbose = 0 为不在标准输出流输出日志信息
#
#     verbose = 1 为输出进度条记录
#     """
#     model = build_model()
#     model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,verbose=0)
#     val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
#     all_scores.append(val_mae)
# #打印出在每个验证集上的平均绝对误差，即验证分数
# print(all_scores)
# #打印出平均验证分数
# print(np.mean(all_scores))
# end = end_time()
# print("总用时：",end - start)

#上面使用的是evaluate()返回验证结果，也就是说每个验证集在fit过后验证一次
"""下面不用evaluate()验证，直接在fit里面提供验证数据，这样每一个epoch都会验证一次，那么每个验证集验证了epochs次，接下来可以
把这么多的验证数据绘制成图像，用来跟踪模型状态
"""
#把前面的代码注释掉，重写上面的for循环
all_mae_history = []
for i in range(k):
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate((train_data[:i * num_val_samples],
                                         train_data[(i + 1) * num_val_samples:]), axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],
                                         train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,
              validation_data=(val_data,val_targets),batch_size=1,verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_history.append(mae_history)
"""下面获取每个验证集在所有轮次上的mae平均值的列表,随便说一下，这个列表的长度肯定是训练轮次大小，每个元素就是所有验证集在该轮次上的平均值"""
average_mae = [np.mean([x[i] for x in all_mae_history]) for i in range(num_epochs)]
"""
 最重要的是图形变化的趋势。因为靠前轮次和后面的轮次的mae可能数值差距较大，故删去；
 将每个点替换为前面点的移动指数平均值，得到光滑的曲线
"""
smooth_points = []
factor = 0.9
for point in average_mae[10:]:
    if smooth_points:
        previous = smooth_points[-1]
        smooth_points.append(previous*factor + point*(1-factor))
    else:
        smooth_points.append(point)
#绘图
epochs_mae(smooth_points)