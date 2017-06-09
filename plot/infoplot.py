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

colnames = ['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14']

shopinfo = pd.read_csv('../data/shop_info.csv')

uniqueshop = list(paycount['shop_id'].unique())
paycount = pd.merge(paycount, shopinfo,on='shop_id',how='left')
paycount =  paycount.sort_values(by='time_stamp',ascending=False)


df= paycount[['shop_id','time_stamp']].groupby(['shop_id']).min()
df =  df.sort_values(by='time_stamp',ascending=False)

newshop = list(df.iloc[0:50].index.unique())


c=0
for i,shop in enumerate(uniqueshop):
	shopdata = paycount[paycount['shop_id']==shop]
	print i,shop
	# if shop!= 2:continue
	fourteen = shopdata.sort_values(by='time_stamp',ascending =False).iloc[0:14]
	lastday = shopdata[shopdata['time_stamp']==pd.datetime(2016,10,31)]['count'].mean()

	# if lastday >= 20:continue


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

	plt.plot(paycount[paycount['shop_id']==shop]['time_stamp'],paycount[paycount['shop_id']==shop]['count'],marker='o')
	plt.plot(paycount[(paycount['shop_id']==shop) & (paycount['weekday']==1)]['time_stamp'],paycount[(paycount['shop_id']==shop) & (paycount['weekday']==1)]['count'],marker='o',color = 'r')
	plt.plot(pd.date_range(start='20161101',end='20161107') , [avg1 * w1,avg1 * w2,avg1 * w3,avg1 * w4,avg1 * w5,avg1 * w6,avg1 * w0],'g')
	plt.plot(pd.date_range(start='20161108',end='20161114') , [avg1 * w1,avg1 * w2,avg1 * w3,avg1 * w4,avg1 * w5,avg1 * w6,avg1 * w0],'g')
	plt.show()
print c
