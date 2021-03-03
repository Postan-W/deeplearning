import shutil
#shutil.copyfileobj(open('./pkl_files/111.pkl','rb'), open('./pkl_files/new1.pkl', 'wb'))
#移动到同个目录下的效果相当于重命名
#shutil.move("./pkl_files/1.pkl","./pkl_files/111.pkl")

#递归删除文件
shutil.rmtree("C:/Users/15216/Desktop/log")
import os
os.rmdir("C:\Users\15216\Desktop\log".replace("\\","/"))