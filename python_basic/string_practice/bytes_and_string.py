"""
Python 3中只有一种能够保存文本信息的数据类型，就是str（string，字符串）。
它是不可变的序列，保存的是Unicode码位（code point）。
这是与Python2的主要区别，Python2用str表示字节字符串，
这种类型现在在Python 3中用bytes对象来处理（但处理方式并不完全相同）。
"""
#bytes以及可变的bytearray与str不同，只能用字节作为序列值，即0<=x<256范围内的整数
#字节也被单引号、双引号或三引号包围，但必须有一个b或B前缀
print(b"abcd")
print(list(b"abcd"))
"""
从Python 3.0开始，所有没有前缀的字符串都是Unicode。因此，所有用单引号（'）、双引号（"）或成组的3个引号（单引号或双引号）包围且没有前缀的值都表示str数据类型
"""
"""
在Python 2中，Unicode需要有u前缀（例如u"some string"）。从Python 3.3开始，为保证向后兼容，仍然可以使用这个前缀，但它在Python 3中没有任何语法上的意义。
"""
#Unicode字符串中包含无法用字节表示的“抽象”文本。因此，如果Unicode字符串没有被编码为二进制数据的话，是无法保存在磁盘中或通过网络发送的。将字符串对象编码为字节序列的方法有两种
"""
利用str.encode(encoding, errors)方法，用注册编解码器（registeredcodec）对字符串进行编码。编解码器由encoding参数指定，默认值为’utf-8'。第二个errors参数指定错误的处理方案，可以取’strict'（默认值）、'ignore'、'replace'、'xmlcharrefreplace’或其他任何注册的处理程序（参见内置codecs模块的文档）
"""
"""
利用bytes(source, encoding, errors)构造函数，创建一个新的字节序列。如果source是str类型，那么必须指定encoding参数，它没有默认值。encoding和errors参数的用法与str.encode()方法中的相同
"""
#用类似方法可以将bytes表示的二进制数据转换成字符串
"""
利用bytes.decode(encoding, errors)方法，用注册编解码器对字节进行解码。这一方法的参数含义及其默认值与str.encode()相同
"""
"""
利用str(source, encoding, error)构造函数，创建一个新的字符串实例。与bytes()构造函数类似，如果source是字节序列的话，必须指定str函数的encoding参数，它没有默认值
"""

