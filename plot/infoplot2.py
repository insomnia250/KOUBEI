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

shopinfo = pd.read_csv('../data/shop_info.csv')

uniqueshop = list(paycount['shop_id'].unique())
paycount = pd.merge(paycount, shopinfo,on='shop_id',how='left')
paycount =  paycount.sort_values(by='time_stamp',ascending=False)


df= paycount[['shop_id','time_stamp']].groupby(['shop_id']).min()
df =  df.sort_values(by='time_stamp',ascending=False)



c=0
for i,shop in enumerate(uniqueshop):
	shopdata = paycount[paycount['shop_id']==shop]
	if not shopdata['cate_2_name'].values[0]=='火锅':continue
	print i,shop,shopdata['cate_2_name'].values[0]
	plt.plot(paycount[paycount['shop_id']==shop]['time_stamp'],paycount[paycount['shop_id']==shop]['count'],marker='o')
	plt.show()
print c
