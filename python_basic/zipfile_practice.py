import zipfile
from zipfile import ZipFile
f = zipfile.ZipFile('C:/Users/15216/Desktop/Desktop.zip')
for file in f.namelist():
    print(file)
    f.extract(file, r'C:/Users/15216/Desktop/target')


"""
 默认值为'r'，表示读已经存在的zip文件，也可以为'w'或'a'，'w'表示新建一个zip文档或覆盖一个已经存在的zip文档
'a'表示将数据附加到一个现存的zip文档中。参数compression表示在写zip文档时使用的压缩方法，它的值可以是zipfile. ZIP_STORED 或zipfile. ZIP_DEFLATED。如果要操作的zip文件大小超过2G，应该将allowZip64设置为True。

ZipFile还提供了如下常用的方法和属性：

ZipFile.getinfo(name)

获取zip文档内指定文件的信息。返回一个zipfile.ZipInfo对象，它包括文件的详细信息。

ZipFile.infolist()

获取zip文档内所有文件的信息，返回一个zipfile.ZipInfo的列表。

ZipFile.namelist()

获取zip文档内所有文件的名称列表。

ZipFile.extract(member[, path[, pwd]])

将zip文档内的指定文件解压到当前目录。参数member指定要解压的文件名称或对应的ZipInfo对象；参数path指定了解析文件保存的文件夹；

参数pwd为解压密码。下面一个例子将保存在程序根目录下的duoduo.zip内的所有文件解压到D:/Work目录：
"""