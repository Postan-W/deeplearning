import pandas as pd
import numpy as np
dates = pd.date_range('20210101',periods=5)

pd1 = pd.DataFrame(np.random.randn(5,5),index=dates,columns=list("ABCDE"))

print(pd1)

pd2 = pd.DataFrame({'A':1,'B':pd.Timestamp('20210101'),'C':pd.Categorical(['test1','test2','test3','test4'])})
print(pd2)
print(pd2.dtypes)
print(pd2.head(1))#看第一行
print(pd2.tail(1))#看倒数第一行
print(pd2.index)#显示行
print(pd2.columns)#显示列

#to_numpy

np1 = pd2.to_numpy()
print(np1,type(np1))
