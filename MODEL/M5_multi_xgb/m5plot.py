#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

SCORE_multixgb = []
SCORE_avg = []
SCORE_multixgbfw = []

paycount = pd.read_csv('../../data/pay_count_cleaned2.csv')
paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
paycount = paycount.sort_values(by=['shop_id','time_stamp'])
paycount['weekday'] = paycount['time_stamp'].dt.dayofweek   #周一对应0

multixgb = pd.read_csv('multixgb_cv.csv')

shopinfo = pd.read_csv('../../data/shop_info.csv')

uniqueshop = list(multixgb['shop_id'].unique())
c=0
multixgbdamnshop ={'shop_id':[],'error':[]}
for i,shop in enumerate(uniqueshop):
	print i,shop
	if shop!=576:continue
	shopdata = paycount[paycount['shop_id']==shop]
	
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
	multixgbpre = multixgb.loc[multixgb['shop_id']==shop].values[0,1:]

	multixgbavg = multixgbpre.mean()
	multixgb_fw = np.array([multixgbavg * w1,multixgbavg *w2,multixgbavg *w3,multixgbavg *w4,multixgbavg * w5,multixgbavg * w6,multixgbavg * w0,
		])

	y = list(target['count'])

	score_avg = (np.abs( (avgpre - y)/ (avgpre + y))).sum()/7.0
	score_multixgb = (np.abs( (multixgbpre - y)/ (multixgbpre + y))).sum()/7.0
	score_multixgbfw = (np.abs( (multixgb_fw - y)/ (multixgb_fw + y))).sum()/7.0


	if (score_multixgb - score_avg)/score_avg>-0.1:
		c+=1
		multixgbdamnshop['shop_id'].append(shop)
		multixgbdamnshop['error'].append((score_multixgb - score_avg)/score_avg)

	print 'avg score:' ,score_avg
	print 'multixgb score:' ,score_multixgb
	print 'multixgbfw score:' ,score_multixgbfw

	SCORE_multixgb.append(score_multixgb)
	SCORE_avg.append(score_avg)
	SCORE_multixgbfw.append(score_multixgbfw)


	plt.plot(paycount[paycount['shop_id']==shop]['time_stamp'],paycount[paycount['shop_id']==shop]['count'],marker='o')
	plt.plot(pd.date_range(start='20161025',end='20161031') ,avgpre ,'g',marker='o')
	plt.plot(pd.date_range(start='20161025',end='20161031') ,multixgbpre ,'r',marker='o')
	# plt.plot(pd.date_range(start='20160717',end='20160730') ,multixgb_fw ,'r',marker='o')
	plt.show()

print c
# multixgbdamnshop = pd.DataFrame(multixgbdamnshop)
# multixgbdamnshop = multixgbdamnshop.sort_values(by='error',ascending=False)
# multixgbdamnshop.to_csv('multixgbdamnshop.csv',index=False)
print 'SCORE_multixgb:', np.mean(SCORE_multixgb)
print 'SCORE_avg:', np.mean(SCORE_avg)
print 'SCORE_multixgbfw:', np.mean(SCORE_multixgbfw)
