import keras2onnx
import time
import onnx
import os
import re
"""
1.对函数的参数进行解释
1.1 model:即是fit后的model;filepath:从文件中加载model的路径,转化为onnx文件的原料
1.2 skeleton：即是模型的所属的框架
1.3 name:你自己给model起的名字，与model.name的值无关
--------------------------------------------------------
2.对onnx模型组件相关信息进行解释
2.1 ir_version,int64: the onnx version assumedby the model.onnx版本
2.2 model_version,int64:模型版本
2.3 doc_string,string:a human-readable documentation for this model.Markdown is allowed.
    模型描述文件，允许使用markdown格式
2.4 metadata_props,map<string,string>:named metadata values,keys should be distinct
    以键值的形式添加元数据，key唯一
"""
#directory是你要保存onnx文件的目录，以"/"结尾
def converted_to_onnx(model,skeleton,name,directory):
    onnx_model = keras2onnx.convert_keras(model,model.name)
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = "wmingzhu"
    meta = onnx_model.metadata_props.add()
    meta.key = "date"
    meta.value = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
    onnx_model.doc_string = "测试使用"
    onnx.save_model(onnx_model, directory + skeleton + "-" + name + ".onnx")    
#此时要引入加载模型的模块
from keras.models import load_model

def converted_to_onnx_from_file(filepath,skeleton,name,directory):
    origin_model = load_model(filepath)
    onnx_model = keras2onnx.convert_keras(origin_model,origin_model.name)
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = "wmingzhu"
    meta = onnx_model.metadata_props.add()
    meta.key = "date"
    meta.value = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
    onnx_model.doc_string = "测试使用"
    onnx.save_model(onnx_model,directory+skeleton+"-"+name+".onnx")

#下面是将序列化过的模型写入文件，作为onnx自带的save_model函数的备选项
def save_onnx_model(filepath,skeleton,name,directory):
    origin_model = load_model(filepath)
    onnx_model = keras2onnx.convert_keras(origin_model,origin_model.name)
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = "wmingzhu"
    meta = onnx_model.metadata_props.add()
    meta.key = "date"
    meta.value = str(time.strftime("%Y.%m.%d-%H.%M.%S"))
    onnx_model.doc_string = "测试使用"
    serialized = onnx_model.SerializeToString()
    with open(directory+skeleton+"-"+name+".onnx","wb") as f:
        f.write(serialized)
