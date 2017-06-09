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

# import train data
countfeature_train = pd.read_csv('../../Feature/count_feature/train_countcl_feature.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_train = pd.merge(countfeature_train, shopinfofeature,how='left',on='shop_id')
# for i in range(13):
# 	df_train.drop(['count_'+str(i)],axis=1,inplace=True)
for i in range(7):
	df_train['fw'+str(i)+'avg1'] = df_train['fw'+str(i)]*df_train['avg1']
	df_train['fw'+str(i)+'avg12'] = df_train['fw'+str(i)]*df_train['avg12']

shop_id_train = df_train['shop_id']
target_train = df_train.ix[:,1:15]
tgweekday_train = df_train.ix[:,15:29]
tgmonth_train = df_train.ix[:,29:43]
tgworkday_train = df_train.ix[:,43:57]
dayopened_train = df_train.ix[:,57:71]
cmft_train = df_train.ix[:,71:]

# import cv data
countfeature_cv = pd.read_csv('../../Feature/count_feature/cv_countcl_feature.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_cv = pd.merge(countfeature_cv, shopinfofeature,how='left',on='shop_id')
# for i in range(13):
# 	df_cv.drop(['count_'+str(i)],axis=1,inplace=True)
for i in range(7):
	df_cv['fw'+str(i)+'avg1'] = df_cv['fw'+str(i)]*df_cv['avg1']
	df_cv['fw'+str(i)+'avg12'] = df_cv['fw'+str(i)]*df_cv['avg12']

shop_id_cv = df_cv['shop_id']
target_cv = df_cv.ix[:,1:15]
tgweekday_cv = df_cv.ix[:,15:29]
tgmonth_cv = df_cv.ix[:,29:43]
tgworkday_cv = df_cv.ix[:,43:57]
dayopened_cv = df_cv.ix[:,57:71]
cmft_cv = df_cv.ix[:,71:]

# import test data
countfeature_test = pd.read_csv('../../Feature/count_feature/test_count_feature.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_test = pd.merge(countfeature_test, shopinfofeature,how='left',on='shop_id')
# for i in range(13):
# 	df_test.drop(['count_'+str(i)],axis=1,inplace=True)
for i in range(7):
	df_test['fw'+str(i)+'avg1'] = df_test['fw'+str(i)]*df_test['avg1']
	df_test['fw'+str(i)+'avg12'] = df_test['fw'+str(i)]*df_test['avg12']

shop_id_test = df_test['shop_id']
tgweekday_test = df_test.ix[:,1:15]
tgmonth_test = df_test.ix[:,15:29]
tgworkday_test = df_test.ix[:,29:43]
dayopened_test = df_test.ix[:,43:57]
cmft_test = df_test.ix[:,57:]


# training
score_xgb = []; score_avg = [];  score_opt = [];  
result_cv = DataFrame(); result_cv['shop_id'] = shop_id_cv
result_test = DataFrame(); result_test['shop_id'] = shop_id_test
for day in range(14):
	# if day==1:break
	X = pd.concat([tgweekday_train.ix[:,day] , tgmonth_train.ix[:,day] ,
		tgworkday_train.ix[:,day] , cmft_train],axis=1)
	X.drop(['min1','min2','tgmonth_'+str(day),'max1','max2'],axis=1,inplace=True)
	# print X.columns
	y = target_train.ix[:,day]
	# y = target_train.mean(1)
	# print y

	X_cv = pd.concat([tgweekday_cv.ix[:,day] , tgmonth_cv.ix[:,day] ,
		tgworkday_cv.ix[:,day] , cmft_cv],axis=1)
	X_cv.drop(['min1','min2','tgmonth_'+str(day),'max1','max2'],axis=1,inplace=True)
	y_cv = target_cv.ix[:,day]


	X_test = pd.concat([tgweekday_test.ix[:,day] , tgmonth_test.ix[:,day] , 
		tgworkday_test.ix[:,day] , cmft_test],axis=1)
	X_test.drop(['min1','min2','tgmonth_'+str(day),'max1','max2'],axis=1,inplace=True)
	# X_test = cmft_test[['avg12','avg1','avg2','davg','ravg','count_13']]#,'median1','median2','median12','sigma']]
	'''
	xgb
	'''
	# model'binary:logistic'
	
	params={
    'booster':'gbtree',
    'objective':'reg:linear',
    'gamma':0.1,
    'max_depth':8,
    #'lambda':250,
    'subsample':0.7,
    # 'colsample_bytree':0.3,
    'min_child_weight':0.3, 
    'eta': 0.04,
    'seed':69,
    'silent':1,
    }

	dtrain = xgb.DMatrix( X, y)

	# # 交叉验证,obj=obj_function
	# bst = xgb.cv(params, dtrain, num_boost_round=500 , nfold=5, feval=eva_function ,verbose_eval=True)

	# # 训练obj=obj_function,
	# num_boost_round = 500  #180
	# watchlist  = [(dtrain,'train')]
	# bst = xgb.train(params, dtrain, num_boost_round)#,evals=watchlist)
	# bst.save_model('./xgb_models/xgb{0}.model'.format(day))
	bst = xgb.Booster(); bst.load_model('./xgb_models/xgb{0}.model'.format(day))
	# # # 打印特征重要性
	# featureColumns = X.columns
	# print_ft_impts(featureColumns,bst)
	#cv
	predcv = bst.predict( xgb.DMatrix( X_cv) )
	result_cv['day_'+str(day)] = list(predcv)

	error = np.abs(result_cv['day_'+str(day)]-y_cv)/(result_cv['day_'+str(day)]+y_cv)
	ers = pd.DataFrame({'shop_id':shop_id_cv,'error':error})

	print ers.sort_values(by='error')

	score_xgb.append( calsore(predcv , y_cv) )
	score_avg.append( calsore(X_cv['avg12'] , y_cv) )
	score_opt.append( calsore(target_cv.mean(1) , y_cv))

	print calsore(predcv , y_cv)
	print calsore(X_cv['avg12'] , y_cv) 
	print calsore(target_cv.mean(1) , y_cv),'\n'
	#test
	predtest = bst.predict( xgb.DMatrix( X_test) )
	result_test['day_'+str(day)] = list(predtest)

	# #1629
	# X_1629 = X[shop_id_train==1629]
	# y_1629 = y[shop_id_train==1629]
	# pre_1629 = bst.predict( xgb.DMatrix( X_1629) )

	# temp=pd.DataFrame({'X_1629':list(X_1629['avg12']),'y_1629':list(y_1629) , 'pre_1629':pre_1629})
	# print temp

result_cv.to_csv('xgb_cv.csv',index=False)
# result_test.to_csv('xgb_test.csv',index=False,header=None)

print 'score_xgb:', np.mean(score_xgb)
print 'score_avg:', np.mean(score_avg)
print 'score_opt:', np.mean(score_opt)

print 'score_xgb:', score_xgb
print 'score_avg:', score_avg
print 'score_opt:', score_opt