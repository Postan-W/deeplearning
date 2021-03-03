#with语句是上下文管理器
"""
任何实现了上下文管理器协议（context manager protocol）的对象都可以用作上下文管理器。
该协议包含两个特殊方法。
• __enter__(self)：
• __exit__(self, exc_type, exc_value, traceback)
简而言之，执行with语句的过程如下：
• 调用__enter__方法。任何返回值都会绑定到指定的as子句。
• 执行内部代码块。
• 调用__exit__方法。
"""
class opened(object):
    def __init__(self,filename):
        self.handle=open(filename)
        print("Resource:%s"%filename)

    def __enter__(self):
        print("[enter%s]: Allocate resource."%self.handle)
        return self.handle#可以返回不同的对象

    def __exit__(self,exc_type,exc_value,exc_trackback):
        print("[Exit %s]: Free resource." %self.handle)
        if exc_trackback is None:
            print("[Exit %s]:Exited without exception."%self.handle)
            self.handle.close()
        else:
            print("[Exit %s]: Exited with exception raised."%self.handle)
        return False # 可以省略，缺省的None也是被看做是False


with opened('../testfiles/1.txt') as fp:
    print(fp.readlines())

"""
注意，多个上下文管理器可以同时使用，如下所示：
with A() as a,B() as b:
等价于
with A() as a:
   with B() as b:
   
"""
