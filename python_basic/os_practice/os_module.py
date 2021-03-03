import os

#获取文件名
url = "http://dafsdfasfasdfa/fasdf/123412.jpg"
file = "C://test/test.txt"
print(os.path.basename(url))#输出123412.jpg
print(os.path.basename(file))#输出test.txt

#获取文件夹下的文件
print(os.listdir("../"))#将返回一个包含上级目录下的所有文件的列表

#分离路径和文件
print(os.path.split(file))#输出(c://test,test.txt)

#移除文件
#os.remove(filepath)

#将路径和文件拼接起来
print(os.path.join("C://test/","test.jpg"))#输出C://test/test.jpg

#将后缀和前面的所有内容分离
print(os.path.splitext("C://test/test.jpg"))#输出结果('C://test/test', '.jpg')