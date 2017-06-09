#coding=utf-8
import pandas as pd 
import csv

paycount_feature = pd.DataFrame()
shop_id = []
day_n = [[] for i in range(14)]
tgweekday_n = [[] for i in range(14)]
tgmonth_n = [[] for i in range(14)]
tgholiday_n = [[] for i in range(14)]
tgworkday_n = [[] for i in range(14)]

avg1=[]; avg2=[];
avgmon = []
avgtue = []
avgwed = []
avgthu = []
avgfri = []
avgsat = []
avgsun = []
count_n = [[] for i in range(14)]

holiday = ['2015-08-20','2015-09-26','2015-09-27','2015-10-01','2015-10-02','2015-10-03',
'2015-10-04','2015-10-05','2015-10-06','2015-10-07','2015-12-24','2015-12-25','2016-01-01',
'2016-01-02','2016-01-03','2016-02-07','2016-02-08','2016-02-09','2016-02-10','2016-02-14',
'2016-02-11','2016-02-12','2016-02-13','2016-04-02','2016-04-03','2016-04-04','2016-04-30',
'2016-05-01','2016-05-02','2016-06-09','2016-06-10','2016-06-11','2016-08-09','2016-09-15',
'2016-09-16','2016-09-17','2016-10-01','2016-10-02','2016-10-03','2016-10-04','2016-10-05',
'2016-10-06','2016-10-07'] 
holiday = pd.to_datetime(holiday)

workday = ['2015-10-08','2015-10-09','2015-10-10','2016-02-06','2016-02-14','2016-06-12',
'2016-09-18','2016-10-08','2016-10-09']
workday = pd.to_datetime(workday)

count = pd.read_csv('../../data/pay_count.csv')
count['time_stamp']=pd.to_datetime(count['time_stamp'])
count = count.sort_values(by='time_stamp')
count['weekday'] = count['time_stamp'].dt.dayofweek   #周一对应0

shoplist = count['shop_id'].unique().tolist()
deltaday = pd.to_timedelta('4 days')  #滑动窗的间隔4天
for i,shop in enumerate(shoplist):
	print i,shop
	# if i==1:break
	shopdata = count[count['shop_id']== shop]
	daystart = shopdata['time_stamp'].min()
	dayend = shopdata['time_stamp'].max() - pd.to_timedelta('27 days') #滑动窗的长度28天

	for day in pd.date_range(start = daystart , end = dayend , freq=deltaday):
		ftday = shopdata[(shopdata['time_stamp']>=day) & (shopdata['time_stamp']<=day +  pd.to_timedelta('13 days') )]
		tgday = shopdata[(shopdata['time_stamp']>=day +  pd.to_timedelta('14 days')) & (shopdata['time_stamp']<=day +  pd.to_timedelta('27 days') )]

		if len(ftday) + len(tgday) == 28:
			shop_id.append(shop)

			#target 14days 
			for i in range(14): day_n[i].append(tgday.iloc[i]['count'])

			#  features
			#  target day feature
			for i in range(14): tgweekday_n[i].append(tgday.iloc[i]['weekday'])
			for i in range(14): tgmonth_n[i].append(tgday.iloc[i]['time_stamp'].month)
			for i in range(14): 
				if tgday.iloc[i]['time_stamp'] in holiday: tgholiday_n[i].append(1)
				else: tgholiday_n[i].append(0)
			for i in range(14): 
				if tgday.iloc[i]['time_stamp'] in workday: tgworkday_n[i].append(1)
				else: tgworkday_n[i].append(0)


			avg1.append( ftday.iloc[0:7]['count'].mean() )
			avg2.append( ftday.iloc[7:14]['count'].mean() )
			avgmon.append( ftday.loc[ftday['weekday']==0]['count'].mean() )
			avgtue.append( ftday.loc[ftday['weekday']==1]['count'].mean() )
			avgwed.append( ftday.loc[ftday['weekday']==2]['count'].mean() )
			avgthu.append( ftday.loc[ftday['weekday']==3]['count'].mean() )
			avgfri.append( ftday.loc[ftday['weekday']==4]['count'].mean() )
			avgsat.append( ftday.loc[ftday['weekday']==5]['count'].mean() )
			avgsun.append( ftday.loc[ftday['weekday']==6]['count'].mean() )
			for i in range(14): count_n[i].append(ftday.iloc[i]['count'])


			# break
# print day_n
# print tgweekday_n
# print tgmonth_n


paycount_feature['shop_id'] = shop_id
for i in range(14):
	paycount_feature['day_'+str(i)] = day_n[i]
for i in range(14):
	paycount_feature['tgweekday_'+str(i)] = tgweekday_n[i]
for i in range(14):
	paycount_feature['tgmonth_'+str(i)] = tgmonth_n[i]
for i in range(14):
	paycount_feature['tgholiday_'+str(i)] = tgholiday_n[i]
for i in range(14):
	paycount_feature['tgworkday_'+str(i)] = tgworkday_n[i]

paycount_feature['avg1'] = avg1
paycount_feature['avg2'] = avg2
paycount_feature['avgmon'] = avgmon
paycount_feature['avgtue'] = avgtue
paycount_feature['avgwed'] = avgwed
paycount_feature['avgthu'] = avgthu
paycount_feature['avgfri'] = avgfri
paycount_feature['avgsat'] = avgsat
paycount_feature['avgsun'] = avgsun
for i in range(14):
	paycount_feature['count_'+str(i)] = count_n[i]

print paycount_feature.info()