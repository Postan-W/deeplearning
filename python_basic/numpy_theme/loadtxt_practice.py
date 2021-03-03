import numpy as np
import pandas as pd
data = np.loadtxt('data_timeseries.txt', delimiter=',')
print(data[:5])
start_date = str(int(data[0,0])) + '-' + str(int(data[0,1]))
end_date = str(int(data[-1,0] + 1)) + '-' + str(int(data[-1,1] % 12 + 1))
print(start_date)
print(end_date)
dates = pd.date_range(start_date, end_date, freq='M')
print(dates)
data_timeseries = pd.Series(data[:, 2], index=dates)
print(data_timeseries[:5])