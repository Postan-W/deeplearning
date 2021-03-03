import torch
import numpy as np
model = torch.load("./models/genderefficientnet-b3.pth",map_location=torch.device('cpu'))
# print(torch.cuda.is_available())
# print(model)

datafile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\data.pkl","rb")
labelsfile = open("C:\\Users\\15216\\Desktop\\data\\dataset\\images1\\jpg2\\labels.pkl","rb")
import pickle
data = pickle.load(datafile)
#data归一化一下
data = np.array(data,dtype="float32")
data /= 255.0

print(model["model"])