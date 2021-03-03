# #在这个模块里使用keras生成的模型
# from keras.models import load_model
# from keras.datasets import reuters
# from utility.vectorization import vectorization
# import numpy as np
# model = load_model("./models/2020.12.15-16.36.37classification_1.h5")
# (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
#
# x_data = vectorization(test_data,10000)
#
# predictions = model.predict(x_data)
#
# print(predictions[0].shape, np.sum(predictions[0]),np.argmax(predictions[0]))
#--------------------------------------------------------------------------------------------
from utility.to_onnx import converted_to_onnx_from_file,save_onnx_model
# converted_to_onnx_from_file("models/2020.12.16-16.09.35keras_classification_2.h5","keras","classification_2")
# save_onnx_model("models/2020.12.16-16.09.35keras_classification_2.h5","keras","classification_2","../../onnx_files/")
converted_to_onnx_from_file("models/2020.12.16-16.09.35keras_classification_2.h5","keras","classification_2","../../onnx_files/")