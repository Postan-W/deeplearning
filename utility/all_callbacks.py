"""
在这里定义各种回调函数列表，有时只是用一个回调函数比如TensorBoard那就可以直接用对应的列表。
有时会使用多个回调函数，那么就把对应的列表唯一的那个回调函数分别作为回调函数列表的元素传入，
比如model.fit(.......,callbacks=[,,,]
"""

import keras

tensorboard = [keras.callbacks.TensorBoard(log_dir='..\\records\\tensorboard',histogram_freq=1, write_grads=True)]

#如果在n轮过后monitor指标依然没改善那就停止训练
early_stopping = [keras.callbacks.EarlyStopping(monitor='acc',patience=1)]

#每轮的权重都保存在指定文件中，如果monitor指标没有改善，那么不需要覆盖模型文件。这就可以始终保存在训练过程中见到的最佳模型
model_check_point = [keras.callbacks.ModelCheckpoint(filepath='..\\records\\best_model\\model.h5',
                                                     monitor='val_loss',save_best_only=True)]

#如果训练进入了loss plateau即损失平台，或者说停滞期，那么增大或减小LR可能有用

reduce_lr_plateau = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10)]
