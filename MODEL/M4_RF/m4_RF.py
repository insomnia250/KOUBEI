#coding=utf-8
from __future__ import division 
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from evalfunc import *
import xgboost as xgb 
from sklearn.ensemble import RandomForestRegressor
import cPickle
from sklearn.model_selection import GridSearchCV
# import train data
countfeature_train = pd.read_csv('../../Feature/count_feature/cl2/cv_countcl_featureTG7.csv')


# ############################修正第二天目标值

weather = pd.read_csv('../../dataset/city_weather.csv',header=None,
	names = ['city_name','time_stamp','t1','t2','weather','wind','windnum'])
weather['time_stamp'] = pd.to_datetime(weather['time_stamp'])
weather.loc[weather['weather']=='大雨' , 'weather']=1
weather.loc[weather['weather']!=1 , 'weather']=0

shopinfo = pd.read_csv('../../data/shop_info.csv')
countfeature_train = pd.merge(countfeature_train,shopinfo[['shop_id','city_name']],how='left',on='shop_id')
raincity = weather[(weather['weather']==1) & (weather['time_stamp']==pd.datetime(2016,10,26))]['city_name'].unique()

# 用最后一周 周二，三，四
for city in raincity:
	countfeature_train.loc[countfeature_train['city_name']==city,'day_1'] = countfeature_train.loc[countfeature_train['city_name']==city,['day_0','day_2','day_6']].mean(1)
countfeature_train.drop(['city_name'],axis=1,inplace=True)
# #########################################

shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_train = pd.merge(countfeature_train, shopinfofeature,how='left',on='shop_id')

# for i in range(13):
# 	df_train.drop(['count_'+str(i)],axis=1,inplace=True)
for i in range(7):
	df_train['fw'+str(i)+'avg1'] = df_train['fw'+str(i)]*df_train['avg1']
	df_train['fw'+str(i)+'avg12'] = df_train['fw'+str(i)]*df_train['avg12']

shop_id_train = df_train['shop_id']
target_train = df_train.ix[:,1:8]
tgweekday_train = df_train.ix[:,8:15]
tgmonth_train = df_train.ix[:,15:22]
tgworkday_train = df_train.ix[:,22:29]
dayopened_train = df_train.ix[:,29:36]
cmft_train = df_train.ix[:,36:]

# import cv data
countfeature_cv = pd.read_csv('../../Feature/count_feature/cl2/cv_countcl_featureTG7.csv');  
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_cv = pd.merge(countfeature_cv, shopinfofeature,how='left',on='shop_id')
# for i in range(13):
# 	df_cv.drop(['count_'+str(i)],axis=1,inplace=True)
for i in range(7):
	df_cv['fw'+str(i)+'avg1'] = df_cv['fw'+str(i)]*df_cv['avg1']
	df_cv['fw'+str(i)+'avg12'] = df_cv['fw'+str(i)]*df_cv['avg12']

shop_id_cv = df_cv['shop_id']
target_cv = df_cv.ix[:,1:8]
tgweekday_cv = df_cv.ix[:,8:15]
tgmonth_cv = df_cv.ix[:,15:22]
tgworkday_cv = df_cv.ix[:,22:29]
dayopened_cv = df_cv.ix[:,29:36]
cmft_cv = df_cv.ix[:,36:]

# import test data
countfeature_test = pd.read_csv('../../Feature/count_feature/test_count_featureTG7.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_test = pd.merge(countfeature_test, shopinfofeature,how='left',on='shop_id')
# for i in range(13):
# 	df_test.drop(['count_'+str(i)],axis=1,inplace=True)
for i in range(7):
	df_test['fw'+str(i)+'avg1'] = df_test['fw'+str(i)]*df_test['avg1']
	df_test['fw'+str(i)+'avg12'] = df_test['fw'+str(i)]*df_test['avg12']

shop_id_test = df_test['shop_id']
tgweekday_test = df_test.ix[:,1:8] 
tgmonth_test = df_test.ix[:,8:15]	
tgworkday_test = df_test.ix[:,15:22]	
dayopened_test = df_test.ix[:,22:29]	
cmft_test = df_test.ix[:,29:]			


avgweekday = ['avgtue','avgwed','avgthu','avgfri','avgsat','avgsun','avgmon']
countday = ['count_0','count_1','count_2','count_3','count_4','count_5',
'count_6','count_7','count_8','count_9','count_10','count_11','count_12','count_13']
fwavg1 = ['fw1avg1','fw2avg1','fw3avg1','fw4avg1','fw5avg1','fw6avg1','fw0avg1']
fwavg12 = ['fw1avg12','fw2avg12','fw3avg12','fw4avg12','fw5avg12','fw6avg12','fw0avg12']
# training
score_RF = []; score_avg = [];  score_opt = [];  score_train=[]
result_cv = DataFrame(); result_cv['shop_id'] = shop_id_cv
result_test = DataFrame(); result_test['shop_id'] = shop_id_test

for day in range(7):
	# if day<=4:continue

	X = pd.concat([tgweekday_train.ix[:,day] , tgmonth_train.ix[:,day] ,
		tgworkday_train.ix[:,day] , cmft_train],axis=1)
	X.drop(['tgmonth_'+str(day)],axis=1,inplace=True)

	y = target_train.ix[:,day]
	# y = target_train.mean(1)
	# print y

	X_cv = pd.concat([tgweekday_cv.ix[:,day] , tgmonth_cv.ix[:,day] ,
		tgworkday_cv.ix[:,day] , cmft_cv],axis=1)
	X_cv.drop(['tgmonth_'+str(day)],axis=1,inplace=True)
	y_cv = target_cv.ix[:,day]


	X_test = pd.concat([tgweekday_test.ix[:,day] , tgmonth_test.ix[:,day] , 
		tgworkday_test.ix[:,day] , cmft_test],axis=1)
	X_test.drop(['tgmonth_'+str(day)],axis=1,inplace=True)
	# for j in range(7):
	# 	X_test['weekday'+str(j)]=0
	# 	X_test.loc[X_test['tgweekday_'+str(day)]==j,'weekday'+str(j)]=1
	# X_test.drop(['tgweekday_'+str(day)],axis=1,inplace=True)
	# for u in range(0,21):
	# 	for v in range(u,21):
	# 		X_test['count_'+str(u)+str(v)] = X_test['count_'+str(u)] * X_test['count_'+str(v)]
	# X_test = cmft_test[['avg12','avg1','avg2','davg','ravg','count_13']]#,'median1','median2','median12','sigma']]
	'''
	RF
	'''
	clf_RF = RandomForestRegressor(n_estimators=100, criterion='mae', max_depth=7, min_samples_split=2, 
						min_samples_leaf=1, min_weight_fraction_leaf=0.0,random_state=69)


	# parameters_for_testing = {
	#    'n_estimators':[50,80,100,200],
	#    'criterion':['mse','mae'],
	#    'max_depth':[3,5,7,9],
	#    'min_samples_split':[2,5,10,15],
	#  }

	# gsearch1 = GridSearchCV(estimator = clf_RF, param_grid = parameters_for_testing,iid=False,verbose=True)
	# gsearch1.fit(X,y)

	# print (gsearch1.grid_scores_)
	# print('best params')
	# print (gsearch1.best_params_)
	# print('best score')
	# print (gsearch1.best_score_)

	clf_RF.fit(X,y)

	# with open('./RF_models/clf_RF.pkl','w') as f:cPickle.dump(clf_RF,f)
	predtrain = clf_RF.predict(X)
	#cv
	predcv = clf_RF.predict(X_cv)
	result_cv['day_'+str(day)] = list(predcv)

	score_train.append( calsore(predtrain , y) )
	score_RF.append( calsore(predcv , y_cv) )
	score_avg.append( calsore(X_cv['avg12'] , y_cv) )
	score_opt.append( calsore(target_cv.mean(1) , y_cv))


	print calsore(predtrain , y)
	print calsore(predcv , y_cv),'\n'
	# print calsore(X_cv['avg12'] , y_cv) 
	# print calsore(target_cv.mean(1) , y_cv),'\n'
	#test
	predtest = clf_RF.predict( X_test) 
	result_test['day_'+str(day)] = list(predtest)



result_cv.to_csv('RF_cv.csv',index=False)
result_test.to_csv('RF_test.csv',index=False,header=None)
print 'score_train:', np.mean(score_train)
print 'score_RF:', np.mean(score_RF)
print 'score_avg:', np.mean(score_avg)
print 'score_opt:', np.mean(score_opt)

print 'score_RF:', score_RF
print 'score_avg:', score_avg
print 'score_opt:', score_opt