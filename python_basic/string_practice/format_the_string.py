#1.%形式
name = "Peter"
id =10
print("my name is %s,id is %d"%(name,id))
print("my name is %(name)s,id is %(id)d"%{"name":name,"id":id})

#2. str.format形式
print('{2} {1} {0}'.format('a','b','c'))
print('{name} {id}'.format(name="Peter",id=10))

#3. f-string形式
name = "Peter"
print(f'你好,{name}')
