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

paycount = pd.read_csv('../../data/pay_count.csv')
paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
paycount = paycount.sort_values(by=['shop_id','time_stamp'])
paycount['weekday'] = paycount['time_stamp'].dt.dayofweek   #周一对应0

XGB = pd.read_csv('xgb_test.csv',header=None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14'])

shopinfo = pd.read_csv('../../data/shop_info.csv')

uniqueshop = list(XGB['shop_id'].unique())
c=0
xgbdamnshop ={'shop_id':[],'error':[]}
for i,shop in enumerate(uniqueshop):
	print i,shop
	if shop!=1824:continue
	shopdata = paycount[paycount['shop_id']==shop]
	
	fourteen = shopdata[(shopdata['time_stamp']>=pd.datetime(2016,10,18)) & (shopdata['time_stamp']<=pd.datetime(2016,10,31))]
	target = shopdata[(shopdata['time_stamp']>=pd.datetime(2016,11,1)) & (shopdata['time_stamp']<=pd.datetime(2016,11,14))]

	avg1 = fourteen['count'].mean()


	w0 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==0)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w1 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==1)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w2 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==2)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w3 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==3)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w4 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==4)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w5 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==5)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()
	w6 = shopdata[(shopdata['shop_id']==shop) & (shopdata['weekday']==6)]['count'].mean()/shopdata[(shopdata['shop_id']==shop)]['count'].mean()

	avgpre = np.array([avg1 * w1,avg1 * w2,avg1 * w3,avg1 * w4,avg1 * w5,avg1 * w6,avg1 * w0,
		avg1 * w1,avg1 * w2,avg1 * w3,avg1 * w4,avg1 * w5,avg1 * w6,avg1 * w0])
	xgbpre = XGB.loc[XGB['shop_id']==shop].values[0,1:]


	score_xgb = (np.abs( (xgbpre - avgpre)/ (xgbpre + avgpre))).sum()/14.0
	if score_xgb>0.1:
		xgbdamnshop['shop_id'].append(shop)
		xgbdamnshop['error'].append(score_xgb)

	plt.plot(paycount[paycount['shop_id']==shop]['time_stamp'],paycount[paycount['shop_id']==shop]['count'],marker='o')
	plt.plot(pd.date_range(start='20161101',end='20161114') ,avgpre ,'g',marker='o')
	plt.plot(pd.date_range(start='20161101',end='20161114') ,xgbpre ,'r',marker='o')
	plt.show()

# xgbdamnshop = pd.DataFrame(xgbdamnshop)
# xgbdamnshop = xgbdamnshop.sort_values(by='error',ascending=False)
# xgbdamnshop.to_csv('xgbdamnshop.csv',index=False)