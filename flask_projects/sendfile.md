关于响应，flask中最简单的响应是字符串。

```
@app.route('/index>')
def index():
    return 'index'
123
```

但是实际上我们常常需要返回json数据，flask中提供了相关的方法：

```
from flask import jsonify
1
@app.route('/index/')
def index():
    data = {
        'name' :'ww'
    }
    data = jsonify(data)
    return data
1234567
```

还有一种更简单的写法

```
@app.route('/index/')
def index():
    return jsonify(name='ww')
123
```

关于request，flask与django有所不同，在django中需要自己设置参数承接数据，flask由框架提供。

```
from flask import request
1
```

request中提供了大量的方法

```
@app.route('/index/',methods=['GET','POST'])
def index():
     # print(request)
        # print(request.url)#请求路径
        # print(request.method)#请求方法
        # print(request.args)
        # print(request.data)#文件(图片等)数据(二进制类型)
        # print(request.headers)#请求头
       # print(request.form)#表单数据
        # print(request.files)  # 文件(类字典类型)
        return  'index'
1234567891011
```

接下来尝试接收前端传来的文件并保存。
这里是通过键值结构从前端传递了一个图片，键名是‘file’，用save（）方法保存。

```
@app.route('/index/',methods=['GET','POST'])
def index():
    data = request.files.get('file')
    data.save('./static/a.png')
    return 'index'
12345
```

也可以这样接收数据并保存

```
@app.route('/index/',methods=['GET','POST'])
def index():
	data = request.data 
    with open('./static/put.png','wb+') as f:
       f.write(data)


    return 'index'
12345678
```