#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

SCORE_xgb = []
SCORE_avg = []
SCORE_xgbfw = []

paycount = pd.read_csv('../../data/pay_count_cleaned2.csv')
paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
paycount = paycount.sort_values(by=['shop_id','time_stamp'])
paycount['weekday'] = paycount['time_stamp'].dt.dayofweek   #周一对应0

XGB = pd.read_csv('xgb_cv.csv')

shopinfo = pd.read_csv('../../data/shop_info.csv')

uniqueshop = list(XGB['shop_id'].unique())
c=0
for i,shop in enumerate(uniqueshop):
	print i,shop
	# if shop!=1629:continue
	shopdata = paycount[paycount['shop_id']==shop]
	
	# fourteen = shopdata[(shopdata['time_stamp']>=pd.datetime(2016,7,3)) & (shopdata['time_stamp']<=pd.datetime(2016,7,16))]
	# target = shopdata[(shopdata['time_stamp']>=pd.datetime(2016,7,17)) & (shopdata['time_stamp']<=pd.datetime(2016,7,23))]
	fourteen = shopdata[(shopdata['time_stamp']>=pd.datetime(2016,10,4)) & (shopdata['time_stamp']<=pd.datetime(2016,10,24))]
	target = shopdata[(shopdata['time_stamp']>=pd.datetime(2016,10,25)) & (shopdata['time_stamp']<=pd.datetime(2016,10,31))]

	
	avg1 = fourteen['count'].mean()

	w0 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==0)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w1 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==1)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w2 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==2)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w3 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==3)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w4 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==4)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w5 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==5)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w6 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==6)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()


	startday = shopdata['time_stamp'].min()
	endday = shopdata['time_stamp'].max()
	daterange = pd.date_range(start = startday , end = endday,freq='D')

	avgpre = np.array([avg1 * w1,avg1 * w2,avg1 *w3,avg1 *w4,avg1 * w5,avg1 * w6,avg1 * w0,
		])
	xgbpre = XGB.loc[XGB['shop_id']==shop].values[0,1:]

	xgbavg = xgbpre.mean()
	xgb_fw = np.array([xgbavg * w1,xgbavg *w2,xgbavg *w3,xgbavg *w4,xgbavg * w5,xgbavg * w6,xgbavg * w0,
		])

	y = list(target['count'])
	print y
	score_avg = (np.abs( (avgpre - y)/ (avgpre + y))).sum()/7.0
	score_xgb = (np.abs( (xgbpre - y)/ (xgbpre + y))).sum()/7.0
	score_xgbfw = (np.abs( (xgb_fw - y)/ (xgb_fw + y))).sum()/7.0


	print 'avg score:' ,score_avg
	print 'xgb score:' ,score_xgb
	print 'xgbfw score:' ,score_xgbfw

	SCORE_xgb.append(score_xgb)
	SCORE_avg.append(score_avg)
	SCORE_xgbfw.append(score_xgbfw)


	# plt.plot(paycount[paycount['shop_id']==shop]['time_stamp'],paycount[paycount['shop_id']==shop]['count'],marker='o')
	# plt.plot(pd.date_range(start='20160717',end='20160730') ,avgpre ,'g',marker='o')
	# # plt.plot(pd.date_range(start='20160717',end='20160730') ,xgbpre ,'r',marker='o')
	# plt.plot(pd.date_range(start='20160717',end='20160730') ,xgb_fw ,'r',marker='o')
	# plt.show()


print 'SCORE_xgb:', np.mean(SCORE_xgb)
print 'SCORE_avg:', np.mean(SCORE_avg)
print 'SCORE_xgbfw:', np.mean(SCORE_xgbfw)
