from PIL import Image

img = Image.open("Koala.jpg")

# 旋转方式一
img1 = img.transpose(Image.ROTATE_180)   # 引用固定的常量值
img1.save("r1.jpg")

# 旋转方式二
img2 = img.rotate(90)   # 自定义旋转度数，逆时针
img2 = img2.resize((400, 400))# 改变图片尺寸
img2.save("r2.jpg")
#旋转不会调换图片的宽和高