import tensorflow as tf
import pathlib
print(tf.__version__)
data_root = pathlib.Path.cwd()
print(data_root)

images_root = pathlib.Path("C:\\Users\\25756\\Desktop\\flowers\\jpg2")

images_dirpath_list = list(images_root.glob('*'))
print(images_dirpath_list[0],type(images_dirpath_list[0]))
#上面只是为了测验功能，现在获取所有图片的路径
all_image_paths = list(images_root.glob('*/*'))
print(all_image_paths[0:3],type(all_image_paths[0]))

#将图片路径转为字符串形式
all_image_paths = [str(path) for path in all_image_paths]
print(all_image_paths[0],type(all_image_paths[0]))
#获取图片类别，即每张图片所在文件夹的名称
label_list = [str(item.name) for item in images_root.glob('*')]
print(label_list)
# 将类别编上号,形如'blueBell': 0, 'buttercup': 1, 'coltsFoot': 2的形式
label_to_index = dict((name,tag) for tag,name in enumerate(label_list))
print(label_to_index,type(label_to_index))
#获取所有图片的类标
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths ]
print(all_image_labels)
#
# #构建图片路径数据集
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
print(path_ds)
# 使用AUTOTUNE自动调节管道参数
AUTOTUNE = tf.data.experimental.AUTOTUNE

#构建图片数据的数据集
from local_utilities.images_process import load_and_process
images_ds =path_ds.map(load_and_process,num_parallel_calls=AUTOTUNE)
# 构建类标数据的数据集

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels,tf.int64))
#将图片和类标压缩为成对形式
image_label_ds = tf.data.Dataset.zip((images_ds,label_ds))

print(images_ds)
print(label_ds)
print(image_label_ds)

#绘出几张图片
from local_utilities import visualization_matplotlib
visualization_matplotlib.images1(image_label_ds)


