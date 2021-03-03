"""
PIL(Python Image Library)是python的第三方图像处理库，PIL的功能非常的强大，几乎被认定是Python的官方图像处理库了。

由于PIL仅支持到python2.7于是一群志愿者在PIL的基础上创建了兼容的版本，名字叫Pillow，支持最新的python3，而且扩容了很多特性，所以在python3我们可以直接安装Pillow。

我们可以去官网查看它的资料：http://effbot.org/。

它可以做的事情：

图像归档(Image Archives)。PIL非常适合于图像归档以及图像的批处理任务。你可以使用PIL创建缩略图，转换图像格式，打印图像等等。
图像展示(Image Display)。PIL较新的版本支持包括Tk PhotoImage，BitmapImage还有Windows DIB等接口。PIL支持众多的GUI框架接口，可以用于图像展示。
图像处理(Image Processing)。PIL包括了基础的图像处理函数，包括对点的处理，使用众多的卷积核(convolution kernels)做过滤(filter),还有颜色空间的转换。PIL库同样支持图像的大小转换，图像旋转，以及任意的仿射变换。PIL还有一些直方图的方法，允许你展示图像的一些统计特性。这个可以用来实现图像的自动对比度增强，还有全局的统计分析等。

"""