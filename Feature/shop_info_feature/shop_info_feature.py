#coding=utf-8
import pandas as pd 
import csv

shop_info = pd.read_csv('../../data/shop_info.csv')
topcity = shop_info['city_name'].value_counts().index
topcate3 = shop_info['cate_3_name'].value_counts().index



shop_info['city'] = 'others'
shop_info['cate3'] = 'others'
shoplist = list(shop_info['shop_id'].unique())
for i,shop in enumerate(shoplist):
	print i
	if shop_info.loc[shop_info['shop_id']==shop]['city_name'].values[0] in topcity:
		shop_info.loc[shop_info['shop_id']==shop,'city'] = shop_info.loc[shop_info['shop_id']==shop]['city_name'].values[0]
	if shop_info.loc[shop_info['shop_id']==shop]['cate_3_name'].values[0] in topcate3:
		shop_info.loc[shop_info['shop_id']==shop,'cate3'] = shop_info.loc[shop_info['shop_id']==shop]['cate_3_name'].values[0]

print shop_info['city'].value_counts()
print pd.get_dummies(shop_info['city'],prefix='city')

print shop_info['cate3'].value_counts()
print pd.get_dummies(shop_info['cate3'],prefix='cate3')

shopinfo = pd.concat([shop_info.loc[:,['shop_id','per_pay','score','comment_cnt','shop_level']],
	pd.get_dummies(shop_info['city'],prefix='city') , pd.get_dummies(shop_info['cate3'],prefix='cate3')],axis=1)

print shopinfo.info()


# '''
# 周期系数
# '''
# fw0=[]
# fw1=[]
# fw2=[]
# fw3=[]
# fw4=[]
# fw5=[]
# fw6=[]
# paycount = pd.read_csv('../../data/pay_count.csv')
# paycount['time_stamp']=pd.to_datetime(paycount['time_stamp'])
# paycount['weekday'] = paycount['time_stamp'].dt.dayofweek   #周一对应0
# paycount = paycount[(paycount['time_stamp']< pd.datetime(2016,10,1)) | (paycount['time_stamp']> pd.datetime(2016,10,7))]

# for shop in shoplist:
# 	shopdata = paycount[paycount['shop_id']==shop]
# 	w0 = shopdata[shopdata['weekday']==0]['count'].mean()/shopdata['count'].mean()
# 	w1 = shopdata[shopdata['weekday']==1]['count'].mean()/shopdata['count'].mean()
# 	w2 = shopdata[shopdata['weekday']==2]['count'].mean()/shopdata['count'].mean()
# 	w3 = shopdata[shopdata['weekday']==3]['count'].mean()/shopdata['count'].mean()
# 	w4 = shopdata[shopdata['weekday']==4]['count'].mean()/shopdata['count'].mean()
# 	w5 = shopdata[shopdata['weekday']==5]['count'].mean()/shopdata['count'].mean()
# 	w6 = shopdata[shopdata['weekday']==6]['count'].mean()/shopdata['count'].mean()

# 	fw0.append(w0)
# 	fw1.append(w1)
# 	fw2.append(w2)
# 	fw3.append(w3)
# 	fw4.append(w4)
# 	fw5.append(w5)
# 	fw6.append(w6)

# fw = pd.DataFrame({'shop_id':shoplist,'fw0':fw0,'fw1':fw1,'fw2':fw2,'fw3':fw3,'fw4':fw4,'fw5':fw5,'fw6':fw6})
# shopinfo = pd.merge(shopinfo,fw,how='left',on='shop_id')

# shopinfo = fw
# print shopinfo.info()
# shopinfo.to_csv('shop_info_feature.csv',index=False,encoding='utf-8')