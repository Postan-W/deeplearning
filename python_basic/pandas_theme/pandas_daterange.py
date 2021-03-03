import pandas as pd
import numpy as np
"""
periods：固定时期，取值为整数或None

freq：日期偏移量，取值为string或DateOffset，默认为'D'

normalize：若参数为True表示将start时间定为00:00:00

name：生成时间索引对象的名称，取值为string或None

closed：可以理解成在closed=None情况下返回的结果中，若closed=‘left’表示在返回的结果基础上，再取左开右闭的结果，若closed='right'表示在返回的结果基础上，再取做闭右开的结果
"""


dates = pd.date_range('20210101', periods=6,name='dates')#等价于(start=20210101,end=20210106,freq='1D,name='dates)
dates2 = pd.date_range('20210101', periods=7,freq="2D")
print(dates)
print(dates2)

dates3 = pd.date_range(start='2017-01-01 08:10:50',periods=10,freq='s',normalize=True)
print(dates3)
dates4 = pd.date_range(start='2017-01-01 08:10:50',periods=10,freq='s',normalize=False)
print(dates4)