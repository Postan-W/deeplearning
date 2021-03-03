from zipfile import ZipFile
import os
import shutil
from fnmatch import fnmatch, fnmatchcase
DL_DIR_PREFIX = "model/model"
with ZipFile("C:\\Users\\15216\\Desktop\\model.zip",'r') as f:
    zips = list(filter(lambda x: x.startswith(DL_DIR_PREFIX), f.namelist()))
    print(zips)
    for element in zips:
        if fnmatch(element,"*.*"):
            f.extract(element,"../../temp_files/")
            entire_path = os.path.join("../../temp_files/",element)
            shutil.move(entire_path,"../../temp_files/")
            shutil.rmtree("../../temp_files/"+"model")
            break
