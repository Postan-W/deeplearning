#sys.path其实是Python的搜索路径集
import sys
print(sys.path)
#如果要动态添加Python路径，可以使用append方法
sys.path.append("D:\\django\\web")
#上面的添加只是临时添加，如果退出当前会话，或者当前的shell，就会消失
