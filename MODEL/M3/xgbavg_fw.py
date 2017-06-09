#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

XGBresult = pd.read_csv('xgb_test.csv',header=None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14'])

XGBresult['avgxgb'] = XGBresult.ix[:,1:15].mean(1)
XGBshoplist = list(XGBresult['shop_id'].unique())


usercount = pd.read_csv('../../data/pay_count.csv')
usercount['time_stamp'] = pd.to_datetime(usercount['time_stamp'])
usercount['weekday'] = usercount['time_stamp'].dt.dayofweek   #周一对应0
shoplist = []
day_1 = []
day_2 = []
day_3 = []
day_4 = []
day_5 = []
day_6 = []
day_7 = []

uniqueshop = list(usercount['shop_id'].unique())
print len(uniqueshop)
df = usercount[(usercount['time_stamp']< pd.datetime(2016,10,1)) | (usercount['time_stamp']> pd.datetime(2016,10,7))]

for i,shop in enumerate(uniqueshop):
	print i,shop
	# if i==2000:break
	shoplist.append(shop)

	shopdata = df[df['shop_id']==shop];#print shopdata
	shopdata = shopdata.sort_values(by='time_stamp',ascending =False).iloc[0:14]   #取最后14天的 ，跳过空值

	# 异常的
	sigma = np.abs(shopdata.iloc[0:7]['count'].mean() - shopdata.iloc[7:14]['count'].mean())/shopdata['count'].mean()
	lastday = shopdata.sort_values(by='time_stamp',ascending =False).iloc[0]['count'] 
	
	#XGB均值
	if shop in XGBshoplist:
		avg_xgb = XGBresult.loc[XGBresult['shop_id']==shop]['avgxgb'].values[0]
		avg1 = avg_xgb
	else:
		avg1 = shopdata['count'].mean()

	if sigma >= 1.0:
		shopdata = df[df['shop_id']==shop];#print shopdata
		shopdata = shopdata.sort_values(by='time_stamp',ascending =False).iloc[0:28]   
		avg1 = shopdata['count'].median()
	
	#lastday


	if shop==23:
		shopdata = df[df['shop_id']==shop];
		shopdata = shopdata.sort_values(by='time_stamp',ascending =False).iloc[30:44]
		avg1 = shopdata['count'].mean()

	

	#周期系数
	# w0 = shopdata[shopdata['weekday']==0]['count'].mean()/avg1
	# w1 = shopdata[shopdata['weekday']==1]['count'].mean()/avg1
	# w2 = shopdata[shopdata['weekday']==2]['count'].mean()/avg1
	# w3 = shopdata[shopdata['weekday']==3]['count'].mean()/avg1
	# w4 = shopdata[shopdata['weekday']==4]['count'].mean()/avg1
	# w5 = shopdata[shopdata['weekday']==5]['count'].mean()/avg1
	# w6 = shopdata[shopdata['weekday']==6]['count'].mean()/avg1

	w0 = df[(df['shop_id']==shop) & (df['weekday']==0)]['count'].mean()/df[(df['shop_id']==shop)]['count'].mean()
	w1 = df[(df['shop_id']==shop) & (df['weekday']==1)]['count'].mean()/df[(df['shop_id']==shop)]['count'].mean()
	w2 = df[(df['shop_id']==shop) & (df['weekday']==2)]['count'].mean()/df[(df['shop_id']==shop)]['count'].mean()
	w3 = df[(df['shop_id']==shop) & (df['weekday']==3)]['count'].mean()/df[(df['shop_id']==shop)]['count'].mean()
	w4 = df[(df['shop_id']==shop) & (df['weekday']==4)]['count'].mean()/df[(df['shop_id']==shop)]['count'].mean()
	w5 = df[(df['shop_id']==shop) & (df['weekday']==5)]['count'].mean()/df[(df['shop_id']==shop)]['count'].mean()
	w6 = df[(df['shop_id']==shop) & (df['weekday']==6)]['count'].mean()/df[(df['shop_id']==shop)]['count'].mean()


	day_1.append(avg1 * w1)   # 2016.11.1 周二
	day_2.append(avg1 * w2)
	day_3.append(avg1 * w3)
	day_4.append(avg1 * w4)
	day_5.append(avg1 * w5)
	day_6.append(avg1 * w6)
	day_7.append(avg1 * w0)

	# plt.plot(df[df['shop_id']==shop]['time_stamp'],df[df['shop_id']==shop]['count'],marker='o',)
	# plt.plot(pd.date_range(start='20161101',end='20161107') , [avg1 * w1,avg1 * w2,avg1 * w3,avg1 * w4,avg1 * w5,avg1 * w6,avg1 * w0],'g')
	# plt.plot(pd.date_range(start='20161101',end='20161107') , [avg2 * w1,avg2 * w2,avg2 * w3,avg2 * w4,avg2 * w5,avg2 * w6,avg2 * w0],'r')
	# plt.show()





result = pd.DataFrame()
result['shop_id'] = np.around(shoplist, decimals=0).astype(int)
result['day_1'] = np.around(day_1, decimals=0).astype(int)
result['day_2'] = np.around(day_2, decimals=0).astype(int)
result['day_3'] = np.around(day_3, decimals=0).astype(int)
result['day_4'] = np.around(day_4, decimals=0).astype(int)
result['day_5'] = np.around(day_5, decimals=0).astype(int)
result['day_6'] = np.around(day_6, decimals=0).astype(int)
result['day_7'] = np.around(day_7, decimals=0).astype(int)
result['day_8'] = np.around(day_1, decimals=0).astype(int)
result['day_9'] = np.around(day_2, decimals=0).astype(int)
result['day_10'] = np.around(day_3, decimals=0).astype(int)
result['day_11'] = np.around(day_4, decimals=0).astype(int)
result['day_12'] = np.around(day_5, decimals=0).astype(int)
result['day_13'] = np.around(day_6, decimals=0).astype(int)
result['day_14'] = np.around(day_7, decimals=0).astype(int)


avgresult = pd.read_csv('../../RULE/RULE3/RULE3_avg1+w365-Oct1_sigma.csv',header=None,
	names=['shop_id','day_1','day_2','day_3','day_4','day_5','day_6','day_7',
	'day_8','day_9','day_10','day_11','day_12','day_13','day_14'])


result['day_8'] = avgresult['day_8']
result['day_9'] = avgresult['day_9']
result['day_10'] = avgresult['day_10']
result['day_11'] = avgresult['day_11']
result['day_12'] = avgresult['day_12']
result['day_13'] = avgresult['day_13']
result['day_14'] = avgresult['day_14']




result.to_csv('XGB2(m3_7+RULE3_7)avgfw+w365-Oct1_sigma.csv',index=False,header=None,encoding = 'utf-8')
print result.describe()
print result.info()
