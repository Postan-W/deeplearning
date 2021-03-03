import pandas as pd
import numpy as np

#用值列表生成 Series 时，Pandas 默认自动生成整数索引
pd1 = pd.Series([1,3,5,np.nan,6,8])#np.nan为空数值
print(pd1)