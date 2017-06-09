#coding=utf-8
# from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

paycount = pd.read_csv('../data/pay_count.csv')
paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
paycount['weekday'] = paycount['time_stamp'].dt.dayofweek   #周一对应0


shoplist = 	list(paycount['shop_id'].unique())

for shop in shoplist:
	shopdata = paycount[(paycount['shop_id']==shop) ]#& (paycount['time_stamp']>pd.datetime(2016,9,10))]

	startday = shopdata['time_stamp'].min()
	endday = shopdata['time_stamp'].max()-pd.to_timedelta('14day')
	daterange = pd.date_range(start = startday , end = endday,freq='D')

	if len(daterange) != len(shopdata)-14:continue

	# dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422, 
	# 6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355, 
	# 10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767, 
	# 12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232, 
	# 13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248, 
	# 9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722, 
	# 11999,9390,13481,14795,15845,15271,14686,11054,10395];

	# dta=pd.Series(dta); dta = dta.astype(float)
	# dta.index = pd.Index(pd.date_range(start=pd.datetime(2016,10,1),periods=90,freq='D' ))
	# print dta.index 
	dta = shopdata['count']; dta.index = shopdata['time_stamp']

	oridta = dta.iloc[0:-14]
	# from statsmodels.tsa.seasonal import seasonal_decompose
	# decomposition = seasonal_decompose(oridta,freq=30)
	# trend = decomposition.trend

	# plt.plot(oridta.index,oridta,marker='o')
	# plt.plot(trend,marker='o')
	# plt.show()

	from statsmodels.tsa.stattools import adfuller
	dftest = adfuller(oridta, autolag='AIC')
	print 'p-value:%.10f' %dftest[1]


	# # grid search for ARMA paras
	# res = sm.tsa.stattools.arma_order_select_ic(oridta, max_ar=7, max_ma=7, ic='bic', trend='c', model_kw={}, fit_kw={})
	# print res.bic_min_order

	# # fitting using minAIC paras
	# arma_mod = sm.tsa.ARMA(oridta,res.bic_min_order).fit(disp=-1)
	# print(arma_mod.aic,arma_mod.bic,arma_mod.hqic)

	# print(sm.stats.durbin_watson(arma_mod70.resid.values))
	# print(sm.stats.durbin_watson(arma_mod66.resid.values))



	# predict_diff = arma_mod.predict(endday+pd.to_timedelta('1day'), '2016-11-14', dynamic=True)
	# predict_diff = arma_mod.predict(pd.datetime(2016,12,30), pd.datetime(2017,1,15), dynamic=True)


	plt.figure()
	# plt.plot(arma_mod.fittedvalues,'r')
	plt.plot(dta,'b')
	# plt.plot(predict_diff.index,predict_diff, 'g')

	w0 = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==0)]['count'].mean()/paycount[(paycount['shop_id']==shop)]['count'].mean()
	w1 = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==1)]['count'].mean()/paycount[(paycount['shop_id']==shop)]['count'].mean()
	w2 = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==2)]['count'].mean()/paycount[(paycount['shop_id']==shop)]['count'].mean()
	w3 = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==3)]['count'].mean()/paycount[(paycount['shop_id']==shop)]['count'].mean()
	w4 = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==4)]['count'].mean()/paycount[(paycount['shop_id']==shop)]['count'].mean()
	w5 = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==5)]['count'].mean()/paycount[(paycount['shop_id']==shop)]['count'].mean()
	w6 = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==6)]['count'].mean()/paycount[(paycount['shop_id']==shop)]['count'].mean()

 

	# RULE5周期不明显的取消波动 500
	wvar = np.var([w1,w2,w3,w4,w5,w6])
	# print wvar
	if wvar < 0.006:continue
	print 'wvar:',wvar

	avg1 = (shopdata.sort_values(by='time_stamp',ascending =False).iloc[14:28])['count'].mean()
	med = (shopdata.sort_values(by='time_stamp',ascending =False).iloc[14:28])['count'].median()

	avgpre = np.array([avg1 * w1,avg1 * w2,avg1 * w3,avg1 * w4,avg1 * w5,avg1 * w6,avg1 * w0,
		avg1 * w1,avg1 * w2,avg1 * w3,avg1 * w4,avg1 * w5,avg1 * w6,avg1 * w0])
	medpre = np.array([med * w1,med * w2,med * w3,med * w4,med * w5,med * w6,med * w0,
		med * w1,med * w2,med * w3,med * w4,med * w5,med * w6,med * w0])
	
	temp = np.array([avgpre,medpre])
	temp_mean = temp.mean()
	mergepre = np.zeros(14)
	mergepre[temp.mean(0)>temp_mean] = temp.max(0)[temp.mean(0)>temp_mean]
	mergepre[temp.mean(0)<=temp_mean] = temp.min(0)[temp.mean(0)<=temp_mean]
	mergepre = 0.5*avgpre + 0.5*medpre

	plt.plot(pd.date_range(start='20161018',end='20161031') ,  avgpre,'k')
	plt.plot(pd.date_range(start='20161018',end='20161031') ,  medpre,'y')
	plt.plot(pd.date_range(start='20161018',end='20161031') ,  mergepre,'r')


	y = dta.iloc[-14:]
	score_avg = (np.abs( (avgpre - y)/ (avgpre + y))).sum()/14.0
	score_med = (np.abs( (medpre - y)/ (medpre + y))).sum()/14.0
	score_merge = (np.abs( (mergepre - y)/ (mergepre + y))).sum()/14.0


	print 'score_avg:{0} ,score_med:{1},score_merge:{2} '.format(score_avg,score_med,score_merge)
	print avg1,med
	plt.show()