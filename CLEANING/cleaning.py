#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


''' 针对统计出的每家商户每天的count值 '''
# # 去掉异常的最大值
# usercount = pd.read_csv('../data/pay_count.csv')
# usercount['time_stamp'] = pd.to_datetime(usercount['time_stamp'])

# countmax = usercount[['shop_id','count']].groupby(['shop_id']).quantile(0.98)

# temp = pd.merge(usercount, countmax, left_on='shop_id', right_index=True,suffixes=('', '_max'))
# temp['count_max'] = temp['count_max']*1.2

# cleanedshops = list(temp.loc[temp['count']>temp['count_max'],:]['shop_id'].unique())

# temp.loc[temp['count']>temp['count_max'],'count'] = temp.loc[temp['count']>temp['count_max'],'count_max']
# countcleaned = temp.drop(['count_max'],axis=1)
# countcleaned.to_csv('../data/pay_count_cleaned.csv',index=False)

# # 最邻近时间的KN填充空值 ， (包括节假日?)
# from sklearn.neighbors import KNeighborsRegressor
# KNclf = KNeighborsRegressor(n_neighbors=28, weights='distance')

# usercount = pd.read_csv('../data/pay_count_cleaned.csv')
# usercount['time_stamp'] = pd.to_datetime(usercount['time_stamp'])
# countcleaned = pd.DataFrame()

# for i,shop in enumerate(usercount['shop_id'].unique()):
# 	print i,shop
# 	shopdata = usercount[usercount['shop_id']==shop]
# 	startday = shopdata['time_stamp'].min()
# 	endday = shopdata['time_stamp'].max()
# 	daterange = pd.date_range(start = startday , end = endday,freq='D')

# 	temp = pd.DataFrame({'time_stamp':daterange});temp['shop_id'] = shop
# 	shopdata = pd.merge(temp,shopdata,on=['shop_id','time_stamp'],how='left')

# 	if len(shopdata[shopdata['count'].isnull()]) > 0:
# 		X = ( (shopdata.loc[shopdata['count'].notnull(),'time_stamp']-startday) / np.timedelta64(1, 'D')).astype(int)
# 		y = shopdata.loc[shopdata['count'].notnull(),'count']
# 		Xnull = ( (shopdata.loc[shopdata['count'].isnull(),'time_stamp']-startday) / np.timedelta64(1, 'D')).astype(int)
		
# 		KNclf.fit(X.reshape(-1,1), y)
# 		shopdata.loc[shopdata['count'].isnull(),'count'] = KNclf.predict(Xnull.reshape(-1,1))

# 	countcleaned = pd.concat([countcleaned,	shopdata],axis=0,ignore_index=True)

# print len(countcleaned), len(usercount)
# print countcleaned.info()
# countcleaned.to_csv('../data/pay_count_cleaned.csv',index=False)
	#'''plot'''
	#print len(cleanedshops)
		# df1 = countcleaned[(countcleaned['time_stamp']< pd.datetime(2016,10,1)) | (countcleaned['time_stamp']> pd.datetime(2016,10,7))]
		# df1 = countcleaned
		# df2 = countcleaned[(countcleaned['time_stamp']< pd.datetime(2016,10,1)) | (countcleaned['time_stamp']> pd.datetime(2016,10,7))]

		# for i,shop in enumerate(cleanedshops):
		# print shop
		# 	# if i==10:break

		# plt.plot(df1[df1['shop_id']==shop]['time_stamp'],df1[df1['shop_id']==shop]['count'],marker='o',color='b')
		# plt.plot(shopdata['time_stamp'],shopdata['count'],marker='o',color='g')
		# plt.show()

# ############################## clean2

# # 空值和holiday用最临近的同weekday填充
# holiday = ['2015-06-20','2015-06-21','2015-08-20','2015-09-03','2015-09-04','2015-09-05','2015-09-26','2015-09-27','2015-10-01','2015-10-02','2015-10-03',
# '2015-10-04','2015-10-05','2015-10-06','2015-10-07','2015-12-24','2015-12-25','2016-01-01',
# '2016-01-02','2016-01-03','2016-02-07','2016-02-08','2016-02-09','2016-02-10','2016-02-14',
# '2016-02-11','2016-02-12','2016-02-13','2016-02-22','2016-04-02','2016-04-03','2016-04-04','2016-04-30','2016-05-20',
# '2016-05-01','2016-05-02','2016-06-09','2016-06-10','2016-06-11','2016-08-09','2016-09-15',
# '2016-09-16','2016-09-17','2016-10-01','2016-10-02','2016-10-03','2016-10-04','2016-10-05',
# '2016-10-06','2016-10-07'] 
# holiday = pd.to_datetime(holiday)

# count = pd.read_csv('../data/pay_count.csv')
# count['time_stamp']=pd.to_datetime(count['time_stamp'])
# count = count.sort_values(by='time_stamp')    #排序  从小到大
# #丢掉节日
# for hld in holiday:
# 	idx = count[count['time_stamp']==hld].index
# 	count.drop(idx,axis=0,inplace=True)



# shoplist = list(count['shop_id'].unique())


# countcleaned = pd.DataFrame()
# c=0
# for i,shop in enumerate(shoplist):
# 	print i,shop
# 	shopdata = count[count['shop_id']==shop]
# 	startday = shopdata['time_stamp'].min()
# 	endday = shopdata['time_stamp'].max()
# 	daterange = pd.date_range(start = startday , end = endday,freq='D')
	
# 	# 直接剔除缺失过多的
# 	if len(shopdata)/len(daterange)<0.7:
# 		continue


# 	temp = pd.DataFrame({'time_stamp':daterange});temp['shop_id'] = shop
# 	shopdata = pd.merge(temp,shopdata,on=['shop_id','time_stamp'],how='left')
# 	shopdata['weekday'] = shopdata['time_stamp'].dt.dayofweek   #周一对应0


# 	naidx = shopdata[shopdata['count'].isnull()].index

# 	#对每个NA
# 	for j,idx in enumerate(naidx):
# 		ts = shopdata.loc[idx]['time_stamp']
# 		wkday = shopdata.loc[idx]['weekday']
		
# 		lpart = shopdata[(shopdata['time_stamp']<ts) & (shopdata['weekday']==wkday) & (shopdata['count'].notnull())]	
# 		rpart = shopdata[(shopdata['time_stamp']>ts) & (shopdata['weekday']==wkday) & (shopdata['count'].notnull())]

# 		appro = []
# 		if lpart.shape[0]!=0:
# 			appro.append( lpart.iloc[-1]['count'] )

# 		if rpart.shape[0]!=0:
# 			appro.append( rpart.iloc[0]['count'] )

# 		shopdata.loc[idx,'count'] = np.mean(appro)

# 	countcleaned = pd.concat([countcleaned,	shopdata],axis=0,ignore_index=True)

# print countcleaned[['shop_id','time_stamp','count']].info()
# countcleaned[['shop_id','time_stamp','count']].to_csv('../data/pay_count_cleaned2.csv',index=False)


#########################################fillna(0)
# # 空值填0
# day924 = pd.datetime(2016,9,24)
# day930 = pd.datetime(2016,9,30)
# day101 = pd.datetime(2016,10,1)
# day107 = pd.datetime(2016,10,7)

# count = pd.read_csv('../data/pay_count.csv')
# count['time_stamp']=pd.to_datetime(count['time_stamp'])
# count = count.sort_values(by='time_stamp')    #排序  从小到大

# shoplist = list(count['shop_id'].unique())


# countcleaned = pd.DataFrame()
# c=0
# for i,shop in enumerate(shoplist):

# 	print i,shop
# 	shopdata = count[count['shop_id']==shop]
# 	startday = shopdata['time_stamp'].min()
# 	endday = shopdata['time_stamp'].max()
# 	daterange = pd.date_range(start = startday , end = endday,freq='D')
	
	
# 	temp = pd.DataFrame({'time_stamp':daterange});temp['shop_id'] = shop
# 	shopdata = pd.merge(temp,shopdata,on=['shop_id','time_stamp'],how='left')
# 	shopdata.fillna(0,inplace=True)


# 	shopdata.loc[ (shopdata['time_stamp']>=day101) & (shopdata['time_stamp']<=day107) , 'count'] = \
# 		shopdata.loc[ (shopdata['time_stamp']>=day924) & (shopdata['time_stamp']<=day930) , 'count'].values
	
# 	countcleaned = pd.concat([countcleaned,	shopdata],axis=0,ignore_index=True)

# print countcleaned[['shop_id','time_stamp','count']].info()
# countcleaned[['shop_id','time_stamp','count']].to_csv('../data/pay_count_cleaned_fill0.csv',index=False)



############################## clean3

# 空值和holiday用最临近的同weekday填充
holiday = ['2015-06-20','2015-06-21','2015-08-20','2015-09-03','2015-09-04','2015-09-05','2015-09-26','2015-09-27','2015-10-01','2015-10-02','2015-10-03',
'2015-10-04','2015-10-05','2015-10-06','2015-10-07','2015-12-24','2015-12-25','2016-01-01',
'2016-01-02','2016-01-03','2016-02-07','2016-02-08','2016-02-09','2016-02-10','2016-02-14',
'2016-02-11','2016-02-12','2016-02-13','2016-02-22','2016-04-02','2016-04-03','2016-04-04','2016-04-30','2016-05-20',
'2016-05-01','2016-05-02','2016-06-09','2016-06-10','2016-06-11','2016-08-09','2016-09-15',
'2016-09-16','2016-09-17','2016-10-01','2016-10-02','2016-10-03','2016-10-04','2016-10-05',
'2016-10-06','2016-10-07'] 
holiday = pd.to_datetime(holiday)

count = pd.read_csv('../data/pay_count.csv')
count['time_stamp']=pd.to_datetime(count['time_stamp'])
count = count.sort_values(by='time_stamp')    #排序  从小到大
#丢掉节日
for hld in holiday:
	idx = count[count['time_stamp']==hld].index
	count.drop(idx,axis=0,inplace=True)



shoplist = list(count['shop_id'].unique())


countcleaned = pd.DataFrame()
c=0
for i,shop in enumerate(shoplist):
	print i,shop
	shopdata = count[count['shop_id']==shop]
	startday = shopdata['time_stamp'].min()
	endday = pd.datetime(2016,10,31)
	daterange = pd.date_range(start = startday , end = endday,freq='D')
	# 直接剔除缺失过多的
	if len(shopdata[(shopdata['time_stamp']>=pd.datetime(2016,10,1)) & (shopdata['time_stamp']<=pd.datetime(2016,10,31))])<20:
		continue
	c+=1

	temp = pd.DataFrame({'time_stamp':daterange});temp['shop_id'] = shop
	shopdata = pd.merge(temp,shopdata,on=['shop_id','time_stamp'],how='left')
	shopdata['weekday'] = shopdata['time_stamp'].dt.dayofweek   #周一对应0


	naidx = shopdata[shopdata['count'].isnull()].index

	#对每个NA
	for j,idx in enumerate(naidx):
		ts = shopdata.loc[idx]['time_stamp']
		wkday = shopdata.loc[idx]['weekday']
		
		lpart = shopdata[(shopdata['time_stamp']<ts) & (shopdata['weekday']==wkday) & (shopdata['count'].notnull())]	
		rpart = shopdata[(shopdata['time_stamp']>ts) & (shopdata['weekday']==wkday) & (shopdata['count'].notnull())]

		appro = []
		if lpart.shape[0]!=0:
			appro.append( lpart.iloc[-1]['count'] )

		if rpart.shape[0]!=0:
			appro.append( rpart.iloc[0]['count'] )

		shopdata.loc[idx,'count'] = np.mean(appro)

	countcleaned = pd.concat([countcleaned,	shopdata],axis=0,ignore_index=True)

print countcleaned[['shop_id','time_stamp','count']].info()
countcleaned[['shop_id','time_stamp','count']].to_csv('../data/pay_count_cleaned3.csv',index=False)

