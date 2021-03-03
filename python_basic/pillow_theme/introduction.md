一、PIL介绍
PIL中所涉及的基本概念有如下几个：通道（bands）、模式（mode）、尺寸（size）、坐标系统（coordinate system）、调色板（palette）、信息（info）和滤波器（filters）

1、 通道
每张图片都是由一个或者多个数据通道构成。PIL允许在单张图片中合成相同维数和深度的多个通道。

以RGB图像为例，每张图片都是由三个数据通道构成，分别为R、G和B通道。而对于灰度图像，则只有一个通道。



2、 模式（mode）
图像的模式定义了图像的类型和像素的位宽。当前支持如下模式：


1：1位像素，表示黑和白，但是存储的时候每个像素存储为8bit。
L：8位像素，表示黑和白。 ****
P：8位像素，使用调色板映射到其他模式。
RGB：3x8位像素，为真彩色。 ****
RGBA：4x8位像素，有透明通道的真彩色。
CMYK：4x8位像素，颜色分离。
YCbCr：3x8位像素，彩色视频格式。
I：32位整型像素。
F：32位浮点型像素。
PIL也支持一些特殊的模式，包括RGBX（有padding的真彩色）和RGBa（有自左乘alpha的真彩色）

3、 尺寸
通过size属性可以获取图片的尺寸。这是一个二元组，包含水平和垂直方向上的像素数。

4、 坐标系统(从左上角开始计算)
PIL使用笛卡尔像素坐标系统，坐标(0，0)位于左上角。注意：坐标值表示像素的角；位于坐标（0，0）处的像素的中心实际上位于（0.5，0.5）。

坐标经常用于二元组（x，y）。长方形则表示为四元组，前面是左上角坐标。例如，一个覆盖800x600的像素图像的长方形表示为（0，0，800，600）。

5、 调色板
调色板模式 ("P")使用一个颜色调色板为每个像素定义具体的颜色值

6、 信息
使用info属性可以为一张图片添加一些辅助信息。这个是字典对象。加载和保存图像文件时，多少信息需要处理取决于文件格式。



7、 滤波器（素描等）


对于将多个输入像素映射为一个输出像素的几何操作，PIL提供了4个不同的采样滤波器：

NEAREST：最近滤波。从输入图像中选取最近的像素作为输出像素。它忽略了所有其他的像素。
BILINEAR：双线性滤波。在输入图像的2x2矩阵上进行线性插值。注意：PIL的当前版本，做下采样时该滤波器使用了固定输入模板。
BICUBIC：双立方滤波。在输入图像的4x4矩阵上进行立方插值。注意：PIL的当前版本，做下采样时该滤波器使用了固定输入模板。
ANTIALIAS：平滑滤波。这是PIL 1.1.3版本中新的滤波器。对所有可以影响输出像素的输入像素进行高质量的重采样滤波，以计算输出像素值。在当前的PIL版本中，这个滤波器只用于改变尺寸和缩略图方法。
注意：在当前的PIL版本中，ANTIALIAS滤波器是下采样（例如，将一个大的图像转换为小图）时唯一正确的滤波器。BILIEAR和BICUBIC滤波器使用固定的输入模板，用于固定比例的几何变换和上采样是最好的。



二、Image方法 常用方法



img = Image.open("1.png")  #获取图片句柄
img.show()                             #打开图片
img.save("t1.jpg")                  #保存图片，保存名字为t1.jpg
img.format                             #img的格式信息（这里是jpg）
img.rotate                             #图片翻转例如；img3 = img.rotate(90) #图片旋转90度

img.resize                            #更改图片大小（缩放等）

例如：

img2 = img.resize((200,200)) #存放元祖
img2.show()
img.mode                            #图像的模式（这里是RGB），（上面的2、 模式（mode））

img1 = img.convert('P')      #转换图像模式（锐化、复古等）
img.size                               #图像的尺寸，按照像素数计算，它的返回值为宽度和高度的二元组（width, height），如(232, 153)
img.info                               #存储图像相关数据的字典。
img.palette                         #颜色调色板表格。如果图像的模式是“P”，则返回ImagePalette类的实例；否则，将为None。
Image.new                          #使用给定的变量mode和size生成新的图像。

例如：

例如：（生成一个图像模式未RGB，全身为红色的128*128像素的正方形）
n_im= Image.new("RGB", (128, 128), "#FF0000")


img.crop                                 #从当前的图像中返回一个矩形区域的拷贝。变量box是一个四元组，定义了左、上、右和下的像素坐标。

例如：



img = Image.open("1.jpg")
box = (300, 100, 700, 700)          ##确定拷贝区域大小
region = img.crop(box)                   ##将im表示的图片对象拷贝到region中，大小为box
region.show()


img.paste                                #将一张图粘贴到另一张图像上（覆盖）。变量box或者是一个给定左上角的2元组，或者是定义了左，上，右和下像素坐标的4元组，或者为空（与（0，0）一样）。如果给定4元组，被粘贴的图像的尺寸必须与区域尺寸一样。如果模式不匹配，被粘贴的图像将被转换为当前图像的模式。

例如：

复制代码
img = Image.open("1.jpg")
box=[0,0,100,100]
im_crop = img.crop(box)
print(im_crop.size,im_crop.mode)
img.paste(im_crop, (100,100))             ##(100,100,0,0)
img.paste(im_crop, (400,400,500,500))
img.show()
复制代码
img.filter                               #返回一个使用给定滤波器处理过的图像的拷贝（图像的浮雕等）。具体参考图像滤波在ImageFilter 模块的应用，在该模块中，预先定义了很多增强滤波器，可以通过filter( )函数使用

例子：

复制代码
img = Image.open("1.jpg")
bluF = img.filter(ImageFilter.BLUR)                ##均值滤波
conF = img.filter(ImageFilter.CONTOUR)             ##找轮廓
edgeF = img.filter(ImageFilter.FIND_EDGES)         ##边缘检测
img.show()
bluF.show()
conF.show()
edgeF.show()
复制代码
img.point                               #返回给定查找表对应的图像像素值的拷贝。变量table为图像的每个通道设置256个值。如果使用变量function，其对应函数应该有一个参数。这个函数将对每个像素值使用一次，结果表格将应用于图像的所有通道。
如果图像的模式为“I（整数）”或者“F（浮点）”，用户必须使用function方式，function必须按照下面的格式：
argument * scale+ offset
例如：
out = im.point(lambda i: i * 1.2 + 10)
用户可以省略变量scale和offset。

img.point例子：

img = Image.open("1.jpg")
im_point = img.point(lambda x:x*1.3+5)
im_point.show()


img.split                        #返回当前图像各个通道组成的一个元组。例如，分离一个“RGB”图像将产生三个新的图像，分别对应原始图像的每个通道（红，绿，蓝）。

例子：

img = Image.open("1.jpg")
r,g,b = img.split()
print(r,g,b)
b.save('b.png')         #无颜色图片
print(b.getpixel((1,3)))  #72


三、其它方法（画图等）
Draw #画图

例子：

复制代码
import PIL.Image as image
import PIL.ImageDraw as draw     #画图
import PIL.ImageFont as imgfont  #字体
img1 = image.open("1.jpg")
img1.show()
img = draw.Draw(img1)
img.point((100,100),fill="red")#画点
img1.show()
img.rectangle((30,30,100,100),outline="red")#画矩形
img1.show()
img.line((20,10,100,120),fill="red",width=2) #画线
img1.show()
font = imgfont.truetype("font.ttf",40) #设置字体和字的大小
img.text((50,50),text="text",fill="red",font=font)#图片中添加文本（"text"）
img1.show()
img.chord((30,50,100,120),100,200,outline="blue") #画圆弧
img1.show()
复制代码




 四、PIL验证码
复制代码
import PIL.Image as image
import PIL.ImageDraw as draw
import PIL.ImageFont as imgfont
import PIL.ImageFilter as ifr
import random
font = imgfont.truetype("font.ttf",60)  #字体

w = 240
h = 120
def randchar():
    '''生成随机字母'''
    return chr(random.randint(65,90))
# print(randchar())

def b_color():
    '''生成随机背景色'''
    return (random.randint(64,255),
            random.randint(64,255),
            random.randint(64,255))

def f_color():
    '''生成随机前景色'''
    return (random.randint(32,128),
            random.randint(32,128),
            random.randint(32,128))

def img():
    return image.new("RGB",(w,h),(255,255,255))

if __name__ == '__main__':
    img = img()
    image = draw.Draw(img)
    for x in range(w):
        for y in range(h):
            image.point((x,y),fill=b_color())
    for i in range(4):
        image.text((60*i+10,30),text=randchar(),fill=f_color(),font=font)
    img.show()
复制代码