import tensorflow as tf

def load_and_process(path):
    image = tf.io.read_file(path)
    #将图片解码为三维矩阵
    image = tf.image.decode_jpeg(image,channels=3)
    print(image)
    #将图片格式统一
    image = tf.image.resize(image,[192,192])
    #对每个像素点的RGB做归一化处理
    image /= 255.0

    return image
