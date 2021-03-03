"""
Flask关于request一些方法和属性的整理
前提：基于纯后端服务，

post 请求 (Content-Type: application/json，)
1.获取未经处理过的原始数据而不管内容类型,如果数据格式是json的，则取得的是json字符串，排序和请求参数一致

c = request.get_data()
2.将请求参数做了处理，得到的是字典格式的，因此排序会打乱依据字典排序规则

c =request.get_json()
3.可以获取未经处理过的原始数据，如果数据格式是json的，则取得的是json字符串，排序和请求参数一致

c = request.data
4.将请求参数做了处理，得到的是字典格式的，因此排序会打乱依据字典排序规则

c = request.json
ps： 刚开始使用的时候以为是一个方法这样调用request.json()然后报错如下：

     Content-Type: application/json时报错'dict' object is not callable

     原来是个属性，因此这样使用request.json,就能正常使用了



我个人做flask取post请求参数一般都是这样用：

a = request.json['a']


get请求 (Content-Type: application/json，)　　
request.args.get('key')  #可以获取到单个的值，

requestValues = request.args  #可以获取get请求的所有参数返回值是ImmutableMultiDict类型,

requestValues.to_dict()  #将获得的参数转换为字典


我个人做flask取get请求参数一般都是这样用：

a = request.args.get('a')


复制代码
#请求头信息
request.headers
#请求方法
request.method
#请求url
request.url
复制代码

"""