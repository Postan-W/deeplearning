"""
1.python获得一些有关系统的各种信息
2.environ是一个字符串所对应环境的映像对象,例如linux下environ[‘HOME’]就代表了当前这个用户的主目录。

3.罗列一些key
windows：

os.environ[‘HOMEPATH’]:当前用户主目录。

os.environ[‘TEMP‘]:临时目录路径。

os.environ[PATHEXT’]:可执行文件。

os.environ[‘SYSTEMROOT‘]:系统主目录。

os.environ[‘LOGONSERVER’]:机器名。

os.environ[‘PROMPT’]:设置提示符。

linux：

os.environ[‘USER‘]:当前使用用户。

os.environ[‘LC_COLLATE’]:路径扩展的结果排序时的字母顺序。

os.environ[‘SHELL’]:使用shell的类型。

os.environ[‘LAN’]:使用的语言。

os.environ[‘SSH_AUTH_SOCK‘]:ssh的执行路径。
"""
import os
#查看environ得key
print(os.environ.keys())
#查看当前用户得主目录
print(os.environ['HOMEPATH'])
#os.environ['PATH']保存的是path下的值，也就是可执行程序得路径，比如python.exe
print(os.environ['PATH'])
#此外可使用os.environ["PATH"] += "新系统环境值"，来添加新环境变量
os.environ["PATH"] += "D:\\ENVS"
#通过上面的方式添加也只是临时性添加，退出会话后自然恢复原来的值
"""
通过export PATH=$PATH:/NewDir可以临时设置系统环境变量，如果要长久生效，必须将export语句放置在/etc/profile文件或者~/.bashrc文件中
"""
os.environ['XQUERY_ADDR'] = "http://"
print(os.environ.get('XQUERY_ADDR'))
