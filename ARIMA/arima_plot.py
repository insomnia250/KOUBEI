#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


paycount = pd.read_csv('../data/pay_count.csv')
paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
paycount = paycount.sort_values(by=['shop_id','time_stamp'])
paycount['weekday'] = paycount['time_stamp'].dt.dayofweek   #周一对应0

ARIMA = pd.read_csv('ARIMA_oripre77.csv',header = None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14'])

print ARIMA.ix[ARIMA['shop_id']==10,1:].values[0]

shopinfo = pd.read_csv('../data/shop_info.csv')

uniqueshop = list(ARIMA['shop_id'].unique())
c=0
for i,shop in enumerate(uniqueshop):
	# print i,shop
	shopdata = paycount[paycount['shop_id']==shop]

	startday = shopdata['time_stamp'].min()
	endday = shopdata['time_stamp'].max()
	daterange = pd.date_range(start = startday , end = endday,freq='D')

	
	dta = shopdata['count']; dta.index = shopdata['time_stamp']
	oridta = dta
	dftest = adfuller(oridta, autolag='AIC')
	print 'p-value:%.10f' %dftest[1]


	
	plt.plot(paycount[paycount['shop_id']==shop]['time_stamp'],paycount[paycount['shop_id']==shop]['count'],marker='o')
	plt.plot(pd.date_range('20161101','20161114'),ARIMA.ix[ARIMA['shop_id']==shop,1:].values[0],marker='o')
	plt.show()
print c