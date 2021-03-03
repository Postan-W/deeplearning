from PIL import Image

img = Image.open("../../testfiles/coltsFoot/image_0881.jpg")

# 获得图像的高和宽
h, w = img.size
print(h,w)
# 获得图像的格式
format = img.format
print(format)
# 获得图像的模式
mo = img.mode
print(mo)
# 将文件重新保存为time.png
#img.save('time', 'png')
# 创建缩略图
#img.thumbnail((50,50),resample=Image.BICUBIC)
#img.show()
# 保存与图像数据相关的字典
dic = img.info
# 验证文件是否损坏，如果损坏回报异常
img.verify()
# 翻转图像
#FLIP_LEFT_RIGHT， FLIP_TOP_BOTTOM，ROTATE_90，ROTATE_180或 ROTATE_270
new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
new_img.show()

#------------------------
# 两张图片混合,两个图像必须具有相同的大小和模式
#Image.blend（image1，image2，alpha）
"""
如果alpha为0.0，则返回第一个图像的副本。如果alpha为1.0，则返回第二个图像的副本。
alpha值没有限制。
如有必要，剪切结果以适应允许的输出范围。
Image.eval（图像，功能） ⇒图像
"""
# 将函数（应该采用一个参数）应用于给定图像中的每个像素
# 色素分离
r, g, b = img.split()
r.show()
g.show()
b.show()
# 旋转图像90度
img1 = img.rotate(90)
img1.show()
# 调整大小
img1 = img.resize((100, 100))
img1.show()
# 将一张图像粘贴到指定位置
#im.paste（图像，方框）
# 调为映像分配存储并从文件（或从源，从延迟操作）加载它
pix  = img.load
print(pix)
# 返回图像的直方图
img_lst = img.histogram()
#im.copy()
#复制图像。如果您希望将内容粘贴到图像中，但仍保留原始图像，请使用此方法。