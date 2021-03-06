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

# csvfile = file('ARIMA_oripre77.csv', 'ab')
# writer = csv.writer(csvfile)

# paycount = pd.read_csv('../data/pay_count.csv')
# paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
# shoplist = []
# day_1 = []; day_8=[]
# day_2 = []; day_9=[]
# day_3 = []; day_10=[]
# day_4 = []; day_11=[]
# day_5 = []; day_12=[]
# day_6 = []; day_13=[]
# day_7 = []; day_14=[]
# uniqueshop = list(paycount['shop_id'].unique())


# # 没有空值的商店
# for i,shop in enumerate(uniqueshop):
# 	if i<=1328:continue
# 	print i,shop
# 	shopdata = paycount[paycount['shop_id']==shop]
# 	startday = shopdata['time_stamp'].min()
# 	endday = shopdata['time_stamp'].max()
# 	daterange = pd.date_range(start = startday , end = endday,freq='D')
# 	if len(daterange) != len(shopdata):continue
	
# 	shoplist.append(shop)
	
# 	dta = shopdata['count']; dta.index = shopdata['time_stamp']
# 	oridta = dta

# 	dftest = adfuller(oridta, autolag='AIC')
# 	# print 'p-value:%.10f' %dftest[1]


# 	# grid search for ARMA paras
# 	res = sm.tsa.stattools.arma_order_select_ic(oridta, max_ar=7, max_ma=5, ic='aic', trend='c', model_kw={}, fit_kw={})
# 	# print res.aic_min_order

# 	# fitting using minAIC paras
# 	try:
# 		arma_mod = sm.tsa.ARMA(oridta,res.aic_min_order).fit(disp=-1)
# 	# print(arma_mod.aic,arma_mod.bic,arma_mod.hqic)
# 	except:
# 		continue
# 	else:
# 		predict_diff = arma_mod.predict(endday+pd.to_timedelta('1day'), '2016-11-14', dynamic=True)
# 		predict_diff = np.around(list(predict_diff), decimals=0).astype(int)
# 		shoppred = [int(shop)]; shoppred.extend(predict_diff)
# 		writer.writerow(shoppred)

# 		# 	plt.figure()
# 		# 	plt.plot(arma_mod.fittedvalues,'r')
# 		# 	plt.plot(oridta,'b')
# 		# 	plt.plot(predict_diff.index,predict_diff, 'g')
# 		# 	plt.show()
# csvfile.close()

ETresult = pd.read_csv('ET_test.csv',header=None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7'])

ETresult['day_8'] = ETresult['day_1']
ETresult['day_9'] = ETresult['day_2']
ETresult['day_10'] = ETresult['day_3']
ETresult['day_11'] = ETresult['day_4']
ETresult['day_12'] = ETresult['day_5']
ETresult['day_13'] = ETresult['day_6']
ETresult['day_14'] = ETresult['day_7']

print ETresult.info()



avgresult = pd.read_csv('../../RULE/RULE3/RULE3_avg1+w365-Oct1_sigma.csv',header=None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14'])

result = pd.concat([ETresult,avgresult],axis=0,ignore_index=True)
result = result.drop_duplicates(['shop_id'],keep = 'first')

result['shop_id'] = result['shop_id'].round().astype(int)
result['day_1'] = result['day_1'].round().astype(int)
result['day_2'] = result['day_2'].round().astype(int)
result['day_3'] = result['day_3'].round().astype(int)
result['day_4'] = result['day_4'].round().astype(int)
result['day_5'] = result['day_5'].round().astype(int)
result['day_6'] = result['day_6'].round().astype(int)
result['day_7'] = result['day_7'].round().astype(int)
result['day_8'] = result['day_8'].round().astype(int)
result['day_9'] = result['day_9'].round().astype(int)
result['day_10'] =result['day_10'].round().astype(int)
result['day_11'] =result['day_11'].round().astype(int)
result['day_12'] =result['day_12'].round().astype(int)
result['day_13'] =result['day_13'].round().astype(int)
result['day_14'] =result['day_14'].round().astype(int)


result = result.sort_values(['shop_id'])
result.index = range(len(result))
avgresult = avgresult.sort_values(['shop_id'])
avgresult.index = range(len(avgresult))

# result['day_8'] = avgresult['day_8']
# result['day_9'] = avgresult['day_9']
# result['day_10'] = avgresult['day_10']
# result['day_11'] = avgresult['day_11']
# result['day_12'] = avgresult['day_12']
# result['day_13'] = avgresult['day_13']
# result['day_14'] = avgresult['day_14']

result.to_csv('m4ET_150_{0}+avg1+w365-oct_sigma.csv'.format(len(ETresult)),index=False,header=None,encoding = 'utf-8')
print result.describe()
print result.info()
