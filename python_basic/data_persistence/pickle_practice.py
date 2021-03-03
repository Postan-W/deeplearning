"""
python的pickle模块提供了一个简答的持久化功能，可以将对象以文件的形式存放在磁盘上。
pickle模块实现了基本的数据序列化和反序列化
通过pickle模块的序列化操作pickle.dump(obj, file, [,protocol])，我们能够将程序中运行的对象信息保存到文件中去，永久存储。
通过pickle模块的反序列化操作pickle.load(file)，我们能从文件中创建上一次程序保存的对象
"""
"""
pickle模块只能在python中使用，python中几乎所有的数据类型（列表，字典，集合，类等）都可以用pickle来序列化，
pickle序列化后的数据，可读性差，人一般无法识别。
"""
"""
存储：pickle.dump(obj, file,[protocol=None])
序列化对象，将对象obj保存到文件file中去。
参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）。
file表示保存到的类文件对象，file必须有write()接口，file可以是一个以’w’打开的文件或者是一个StringIO对象，也可以是任何可以实现write()接口的对象。
"""
import pickle
#dump和load-----------------------------------------------------------------------------
#创建一个字典变量
data = {'a':[1,2,3],'b':('string','abc'),'c':'hello'}
print(data)

#以二进制方式来存储,rb,wb,wrb,ab
pic = open('./pkl_files/1.pkl','wb')

#将字典数据存储为一个pkl文件
pickle.dump(data,pic)
pic.close()

#读取 pickle.load(file)
pic2 = open("./pkl_files/1.pkl",'rb')
data = pickle.load(pic2)
print(data)
print(type(data))

#dumps和loads-----------------------------------------------------------------------------------------
l1 = [1, 2, 3, 4, 5]
t1 = (1, 2, 3, 4, 5)
dic1 = {"k1": "v1", "k2": "v2", "k3": "v3"}

res_l1 = pickle.dumps(l1)
res_t1 = pickle.dumps(t1)
res_dic = pickle.dumps(dic1)
print(res_l1)
print(res_t1)
print(res_dic)
#用loads反序列化
print(pickle.loads(res_l1),type(pickle.loads(res_l1)))
print(pickle.loads(res_t1),type(pickle.loads(res_t1)))
print(pickle.loads(res_dic),type(pickle.loads(res_dic)))


