#这里为项目提供一些图形化工具

import matplotlib.pyplot as plt
#这里函数接收训练过程中产生的history对象中包含的history_dict数据
def  training_and_validation_loss(history_dict):
    #训练损失
    loss_values = history_dict['loss']
    #验证损失
    val_loss_values = history_dict['val_loss']
    #轮次(一个轮次就是所有训练数据数据训练一次，这其中包含了多个batch，而一个batch意味着梯度的一次更新)
    epochs = range(1,len(loss_values) + 1)
    plt.plot(epochs,loss_values,'ro',label="training loss")
    plt.plot(epochs,val_loss_values,'b',label="validation loss")
    plt.title("training and validation loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.show()

#准确度随着训练轮数的变化
def training_and_validation_accuracy(history_dict):
    acc = history_dict['acc']
    val_acc = history_dict['val_acc']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
#绘制每个轮次的mae
def epochs_mae(mae):
    plt.plot(range(1,len(mae)+1),mae)
    plt.xlabel('epochs')
    plt.ylabel('validation mae')
    plt.show()
