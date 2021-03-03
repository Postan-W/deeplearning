import json
"""
json是字符串
json中不存在元组。序列化元组之后元组变列表；不能是集合，序列化集合报错。
序列化支持类型可以进Python官方文件介绍里面有介绍。以后传值就是传一个也要用字典或列表。
"""
#json.loads方法可以将包含了一个JSON数据的str, bytes或者bytearray对象, 转化为一个Python数据结构
print(json.loads("[1,2,3]"),type(json.loads("[1,2,3]")))
#-------------------------------------------------
t1 = (1,2,3,5,7,8,9,0,12,3,4,5,6,7,8,8,9)
t2 = json.dumps(t1)
print(t2,type(t2))
t3 = json.loads(t2)
print(t3,type(t3))

t4 = {0:[1,2,3],1:[4],2:"hello","a":"abcddsaf"}
t5 = json.dumps(t4,indent=1)
print(t5,type(t5))
t6 = json.loads(t5)
print(t6,type(t6))

#load和dump方法是对应文件的操作，可参考pickle的这部分内容------------------------------------
