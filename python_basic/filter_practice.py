name_list = ["zhangyi","zhanger","zhangsan","liyi","lier"]
#filter的第一个参数是操作函数，第二个参数是可迭代对象
zhang =list(filter(lambda x:x.startswith("zhang"),name_list))

print(zhang)