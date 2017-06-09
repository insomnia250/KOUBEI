#coding=utf-8
# from __future__ import print_function
import csv
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller

csvfile = file('ARIMA_diff1pre77.csv', 'ab')
writer = csv.writer(csvfile)

paycount = pd.read_csv('../data/pay_count.csv')
paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
shoplist = []
day_1 = []; day_8=[]
day_2 = []; day_9=[]
day_3 = []; day_10=[]
day_4 = []; day_11=[]
day_5 = []; day_12=[]
day_6 = []; day_13=[]
day_7 = []; day_14=[]
uniqueshop = list(paycount['shop_id'].unique())


# 没有空值的商店
for i,shop in enumerate(uniqueshop):
	# if i==6:break
	print i,shop
	shopdata = paycount[paycount['shop_id']==shop]
	startday = shopdata['time_stamp'].min()
	endday = shopdata['time_stamp'].max()
	daterange = pd.date_range(start = startday , end = endday,freq='D')
	if len(daterange) != len(shopdata):continue
	
	shoplist.append(shop)
	
	dta = shopdata['count']; dta.index = shopdata['time_stamp']
	oridta = dta
	diffdta= dta.diff(1)

	dftest = adfuller(diffdta.iloc[1:], autolag='AIC')
	# print 'p-value:%.10f' %dftest[1]


	# grid search for ARMA paras
	res = sm.tsa.stattools.arma_order_select_ic(diffdta.iloc[1:], max_ar=7, max_ma=5, ic='aic', trend='c', model_kw={}, fit_kw={})
	# print res.aic_min_order

	# fitting using minAIC paras
	try:
		arma_mod = sm.tsa.ARMA(diffdta.iloc[1:],res.aic_min_order).fit(disp=-1)
	# print(arma_mod.aic,arma_mod.bic,arma_mod.hqic)
	except:
		continue
	else:
		predict_diff = arma_mod.predict(endday+pd.to_timedelta('1day'), '2016-11-14', dynamic=True)
		predict_diff = predict_diff.cumsum() + oridta.iloc[-1]

		predict_diff = np.around(list(predict_diff), decimals=0).astype(int)
		shoppred = [int(shop)]; shoppred.extend(predict_diff)
		writer.writerow(shoppred)

		# 	plt.figure()
		# 	plt.plot(arma_mod.fittedvalues,'r')
		# 	plt.plot(oridta,'b')
		# 	plt.plot(predict_diff.index,predict_diff, 'g')
		# 	plt.show()
csvfile.close()

ARIMAresult = pd.read_csv('ARIMA_diff1pre77.csv',header=None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14'])
avgresult = pd.read_csv('../RULE/RULE3/RULE3_avg1+w365-Oct1_sigma.csv',header=None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14'])

result = pd.concat([ARIMAresult,avgresult],axis=0,ignore_index=True)
result = result.drop_duplicates(['shop_id'],keep = 'first')

result.to_csv('diff1ARIMA{0}+avg1+w365-oct_sigma.csv'.format(len(ARIMAresult)),index=False,header=None,encoding = 'utf-8')
# print result.describe()
# print result.info()
