1）导入模块

import requests
2）发送请求的简洁

　　示例代码：获取一个网页（个人github）

import requests

r = requests.get('https://github.com/Ranxf')       # 最基本的不带参数的get请求
r1 = requests.get(url='http://dict.baidu.com/s', params={'wd': 'python'})      # 带参数的get请求
我们就可以使用该方式使用以下各种方法

1   requests.get(‘https://github.com/timeline.json’)                                # GET请求
2   requests.post(“http://httpbin.org/post”)                                        # POST请求
3   requests.put(“http://httpbin.org/put”)                                          # PUT请求
4   requests.delete(“http://httpbin.org/delete”)                                    # DELETE请求
5   requests.head(“http://httpbin.org/get”)                                         # HEAD请求
6   requests.options(“http://httpbin.org/get” )                                     # OPTIONS请求
3）为url传递参数

>>> url_params = {'key':'value'}       #    字典传递参数，如果值为None的键不会被添加到url中
>>> r = requests.get('your url',params = url_params)
>>> print(r.url)
　　your url?key=value
4）响应的内容

复制代码
r.encoding                       #获取当前的编码
r.encoding = 'utf-8'             #设置编码
r.text                           #以encoding解析返回内容。**字符串方式**的响应体，会自动根据响应头部的字符编码进行解码。
r.content                        #以**字节形式（二进制）**返回。字节方式的响应体，会自动为你解码 gzip 和 deflate 压缩。

r.headers                        #以字典对象存储服务器响应头，但是这个字典比较特殊，字典键不区分大小写，若键不存在则返回None

r.status_code                     #响应状态码
r.raw                             #返回原始响应体，也就是 urllib 的 response 对象，使用 r.raw.read()   
r.ok                              # 查看r.ok的布尔值便可以知道是否登陆成功
 #*特殊方法*#
r.json()                         #Requests中内置的JSON解码器，以json形式返回,前提返回的内容确保是json格式的，不然解析出错会抛异常
r.raise_for_status()             #失败请求(非200响应)抛出异常
复制代码
post发送json请求：

1 import requests
2 import json
3  
4 r = requests.post('https://api.github.com/some/endpoint', data=json.dumps({'some': 'data'}))
5 print(r.json())
5）定制头和cookie信息

header = {'user-agent': 'my-app/0.0.1''}
cookie = {'key':'value'}
 r = requests.get/post('your url',headers=header,cookies=cookie) 
data = {'some': 'data'}
headers = {'content-type': 'application/json',
           'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:22.0) Gecko/20100101 Firefox/22.0'}

r = requests.post('https://api.github.com/some/endpoint', data=data, headers=headers)
print(r.text)
6）响应状态码

使用requests方法后，会返回一个response对象，其存储了服务器响应的内容，如上实例中已经提到的 r.text、r.status_code……
获取文本方式的响应体实例：当你访问 r.text 之时，会使用其响应的文本编码进行解码，并且你可以修改其编码让 r.text 使用自定义的编码进行解码。

1 r = requests.get('http://www.itwhy.org')
2 print(r.text, '\n{}\n'.format('*'*79), r.encoding)
3 r.encoding = 'GBK'
4 print(r.text, '\n{}\n'.format('*'*79), r.encoding)
示例代码：

复制代码
1 import requests
2 
3 r = requests.get('https://github.com/Ranxf')       # 最基本的不带参数的get请求
4 print(r.status_code)                               # 获取返回状态
5 r1 = requests.get(url='http://dict.baidu.com/s', params={'wd': 'python'})      # 带参数的get请求
6 print(r1.url)
7 print(r1.text)        # 打印解码后的返回数据
复制代码
运行结果：

/usr/bin/python3.5 /home/rxf/python3_1000/1000/python3_server/python3_requests/demo1.py
200
http://dict.baidu.com/s?wd=python
…………

Process finished with exit code 0
 r.status_code                      #如果不是200，可以使用 r.raise_for_status() 抛出异常
7）响应

r.headers                                  #返回字典类型,头信息
r.requests.headers                         #返回发送到服务器的头信息
r.cookies                                  #返回cookie
r.history                                  #返回重定向信息,当然可以在请求是加上allow_redirects = false 阻止重定向
8）超时

 

r = requests.get('url',timeout=1)           #设置秒数超时，仅对于连接有效


9)会话对象，能够跨请求保持某些参数

s = requests.Session()
s.auth = ('auth','passwd')
s.headers = {'key':'value'}
r = s.get('url')
r1 = s.get('url1') 
10）代理

 

proxies = {'http':'ip1','https':'ip2' }
requests.get('url',proxies=proxies)

------

# Flask中GET和POST请求

#### post 请求 (Content-Type: application/json，)

1.获取未经处理过的原始数据而不管内容类型,如果数据格式是json的，则取得的是json字符串，排序和请求参数一致

```
c = request.get_data()    
```

2.将请求参数做了处理，得到的是字典格式的，因此排序会打乱依据字典排序规则

```
c =request.get_json()
```

3.可以获取未经处理过的原始数据，如果数据格式是json的，则取得的是json字符串，排序和请求参数一致

```
c = request.data
```

4.将请求参数做了处理，得到的是字典格式的，因此排序会打乱依据字典排序规则

```
c = request.json
```

ps： 刚开始使用的时候以为是一个方法这样调用request.json()然后报错如下：

   Content-Type: application/json时报错'dict' object is not callable

   原来是个属性，因此这样使用request.json,就能正常使用了

 

我个人做flask取post请求参数一般都是这样用：

```
a = request.json['a']
```

 

#### get请求 (Content-Type: application/json，)　　

```
request.args.get('key')  #可以获取到单个的值，

requestValues = request.args  #可以获取get请求的所有参数返回值是ImmutableMultiDict类型,

requestValues.to_dict()  #将获得的参数转换为字典
```

 

我个人做flask取get请求参数一般都是这样用：

```
a = request.args.get('a')
```

 

[![复制代码](https://common.cnblogs.com/images/copycode.gif)](javascript:void(0);)

```
#请求头信息
request.headers 
#请求方法
request.method 
#请求url   
request.url
```