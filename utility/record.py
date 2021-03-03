#将程序运行的一些信息记录到文件中，比如深度学习模型训练过程中产生的数据、程序的error数据等
import sys
import time
#第一个参数是要记录的信息，第二个参数是信息保存的目标文件,第三个关键字参数是所在区域
def record(information,file,zone="Beijing"):
    with open(file,'a',encoding='utf-8') as f:
        #输出重定向到目标文件
        sys.stdout = f
        date = time.strftime("%Y/%m/%d")
        timer = time.strftime("%H:%M:%S")
        sys.stdout.writelines(date + "  " + timer + "," + zone + "\n",)
        sys.stdout.writelines(information + "\n\n\n\n\n\n\n\n\n")
        #恢复标准输出
        sys.stdout = sys.__stdout__


record("爱好速度发好了大家啊胜利大街发链接是的话\n费卡说服力阿达防守打法大事发生大发生发送到发vsadcxzcCzcx11","..\\records\\error.txt")

print("标准输出已恢复")

