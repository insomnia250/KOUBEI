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
from sklearn.neural_network import MLPRegressor
import cPickle
from sklearn.model_selection import GridSearchCV
# import train data
countfeature_train = pd.read_csv('../../Feature/count_feature/cl2/train_countcl_featureTG7.csv')
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


# #归一化
import sklearn.preprocessing as preprocessing
ColumnList = cmft_train[['avg1','avg2','avg12','davg','ravg','median1','median2','median12','min1','min2','max1','max2',
		'avgmon','avgtue','avgwed','avgthu','avgfri','avgsat','avgsun','count_0','count_1','count_2','count_3','count_4','count_5',
		'count_6','count_7','count_8','count_9','count_10','count_11','count_12','count_13','fw0avg1','fw0avg12','fw1avg1','fw1avg12',
		'fw2avg1','fw2avg12','fw3avg1','fw3avg12','fw4avg1','fw4avg12','fw5avg1','fw5avg12','fw6avg1','fw6avg12']].columns
for col in ColumnList:
	scaler = preprocessing.StandardScaler()
	cmft_train.loc[:,col] = scaler.fit_transform(cmft_train[col].reshape(-1,1))
	cmft_cv.loc[:,col] = scaler.transform(cmft_cv[col].reshape(-1,1))
	cmft_test.loc[:,col] = scaler.transform(cmft_test[col].reshape(-1,1))



avgweekday = ['avgtue','avgwed','avgthu','avgfri','avgsat','avgsun','avgmon']
countday = ['count_0','count_1','count_2','count_3','count_4','count_5',
'count_6','count_7','count_8','count_9','count_10','count_11','count_12','count_13']
fwavg1 = ['fw1avg1','fw2avg1','fw3avg1','fw4avg1','fw5avg1','fw6avg1','fw0avg1']
fwavg12 = ['fw1avg12','fw2avg12','fw3avg12','fw4avg12','fw5avg12','fw6avg12','fw0avg12']
# training
score_nn = []; score_avg = [];  score_opt = [];  score_train=[]
result_cv = DataFrame(); result_cv['shop_id'] = shop_id_cv
result_test = DataFrame(); result_test['shop_id'] = shop_id_test
for day in range(7):
	# if day==1:break

	X = pd.concat([cmft_train],axis=1)

	X = X[['avg1','avg2','avg12',avgweekday[day],'count_7','count_8','count_9','count_10','count_11','count_12','count_13',
	fwavg1[day] , fwavg12[day] ]]

	# for j in range(7):
	# 	X['weekday'+str(j)]=0
	# 	X.loc[X['tgweekday_'+str(day)]==j,'weekday'+str(j)]=1
	# X.drop(['tgweekday_'+str(day)],axis=1,inplace=True)

	y = target_train.ix[:,day]
	# y = target_train.mean(1)
	# print y

	X_cv = pd.concat([cmft_cv],axis=1)
	X_cv = X_cv[['avg1','avg2','avg12',avgweekday[day],'count_7','count_8','count_9','count_10','count_11','count_12','count_13',
	fwavg1[day] , fwavg12[day] ]]
	# for u in range(0,21):
	# 	for v in range(u,21):
	# 		X_cv['count_'+str(u)+str(v)] = X_cv['count_'+str(u)] * X_cv['count_'+str(v)]
	# for j in range(7):
	# 	X_cv['weekday'+str(j)]=0
	# 	X_cv.loc[X_cv['tgweekday_'+str(day)]==j,'weekday'+str(j)]=1
	# X_cv.drop(['tgweekday_'+str(day)],axis=1,inplace=True)
	y_cv = target_cv.ix[:,day]


	X_test = pd.concat([cmft_test],axis=1)
	X_test = X_test[['avg1','avg2','avg12',avgweekday[day],'count_7','count_8','count_9','count_10','count_11','count_12','count_13',
	fwavg1[day] , fwavg12[day] ]]
	# for j in range(7):
	# 	X_test['weekday'+str(j)]=0
	# 	X_test.loc[X_test['tgweekday_'+str(day)]==j,'weekday'+str(j)]=1
	# X_test.drop(['tgweekday_'+str(day)],axis=1,inplace=True)
	# for u in range(0,21):
	# 	for v in range(u,21):
	# 		X_test['count_'+str(u)+str(v)] = X_test['count_'+str(u)] * X_test['count_'+str(v)]
	# X_test = cmft_test[['avg12','avg1','avg2','davg','ravg','count_13']]#,'median1','median2','median12','sigma']]
	'''
	nn
	'''
	clf_nn = MLPRegressor(activation ='logistic',alpha=0.1,hidden_layer_sizes = (40) ,
							max_iter = 3000, validation_fraction=0,early_stopping =False, random_state =1,
							learning_rate_init=0.003)#,verbose=True


	# parameters_for_testing = {
	#    'activation':['logistic'],
	#    'hidden_layer_sizes':[(25),(30),(35),(40)],
	#    'max_iter':[3000,4000,5000,2000],
	#    'learning_rate_init':[0.0012,0.002,0.003,0.004,0.0035,0.0005],
	#  }

	# gsearch1 = GridSearchCV(estimator = clf_nn, param_grid = parameters_for_testing,iid=False)
	# gsearch1.fit(X,y)

	# print (gsearch1.grid_scores_)
	# print('best params')
	# print (gsearch1.best_params_)
	# print('best score')
	# print (gsearch1.best_score_)

	clf_nn.fit(X,y)

	with open('./nn_models/clf_nn.pkl','w') as f:cPickle.dump(clf_nn,f)
	predtrain = clf_nn.predict(X)
	#cv
	predcv = clf_nn.predict(X_cv)
	result_cv['day_'+str(day)] = list(predcv)

	score_train.append( calsore(predtrain , y) )
	score_nn.append( calsore(predcv , y_cv) )
	# score_avg.append( calsore(X_cv['avg12'] , y_cv) )
	# score_opt.append( calsore(target_cv.mean(1) , y_cv))


	print calsore(predtrain , y)
	print calsore(predcv , y_cv),'\n'
	# print calsore(X_cv['avg12'] , y_cv) 
	# print calsore(target_cv.mean(1) , y_cv),'\n'
	#test
	predtest = clf_nn.predict( X_test) 
	result_test['day_'+str(day)] = list(predtest)

	#1629
	X_1629 = X[shop_id_train==1629]
	y_1629 = y[shop_id_train==1629]
	pre_1629 = clf_nn.predict( X_1629) 

	temp=pd.DataFrame({'X_1629':list(X_1629['avg12']),'y_1629':list(y_1629) , 'pre_1629':pre_1629})
	print temp

# result_cv.to_csv('nn_cv.csv',index=False)
# result_test.to_csv('nn_test.csv',index=False,header=None)
# print 'score_train:', np.mean(score_train)
# print 'score_nn:', np.mean(score_nn)
# # print 'score_avg:', np.mean(score_avg)
# # print 'score_opt:', np.mean(score_opt)

# print 'score_nn:', score_nn
# # print 'score_avg:', score_avg
# # print 'score_opt:', score_opt