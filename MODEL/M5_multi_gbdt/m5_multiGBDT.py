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
from sklearn.ensemble import GradientBoostingRegressor
import cPickle

# import train data
countfeature_train = pd.read_csv('../../Feature/count_feature/train_countcl_featureTG7.csv')
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
countfeature_cv = pd.read_csv('../../Feature/count_feature/cv_countcl_featureTG7.csv');  
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


print set(list(shop_id_test)) - set(list(shop_id_train))

# training
score_gbdt = []; score_avg = [];  score_opt = [];  score_train=[]
result_cv = DataFrame(); result_cv['shop_id'] = shop_id_cv
result_test = DataFrame(); result_test['shop_id'] = shop_id_test

#multi target
from sklearn.multioutput import MultiOutputRegressor

X = pd.concat([ cmft_train],axis=1)  #tgweekday_train ,tgworkday_train,
print X.info()
# for u in range(0,21):
# 	for v in range(u,21):
# 		X['count_'+str(u)+str(v)] = X['count_'+str(u)] * X['count_'+str(v)]
y = target_train
# y = target_train.mean(1)
# print y

#cv
X_cv = pd.concat([cmft_cv],axis=1)   #tgweekday_cv ,tgworkday_cv, 
# for u in range(0,21):
# 	for v in range(u,21):
# 		X_cv['count_'+str(u)+str(v)] = X_cv['count_'+str(u)] * X_cv['count_'+str(v)]
y_cv = target_cv

#test
X_test = pd.concat([ cmft_test],axis=1)  #tgweekday_test,tgworkday_test,
# for u in range(0,21):
# 	for v in range(u,21):
# 		X_test['count_'+str(u)+str(v)] = X_test['count_'+str(u)] * X_test['count_'+str(v)]
# X_test = cmft_test[['avg12','avg1','avg2','davg','ravg','count_13']]#,'median1','median2','median12','sigma']]
'''
gbdt
'''
clf_gbdt = MultiOutputRegressor(GradientBoostingRegressor(learning_rate=0.02,n_estimators=290,max_depth=4,
	subsample=1.0))

clf_gbdt.fit(X,y)


# with open('./gbdt_models/clf_gbdt.pkl','w') as f:cPickle.dump(clf_gbdt,f)
predtrain = clf_gbdt.predict(X)
#cv
predcv = clf_gbdt.predict(X_cv)
#test
predtest = clf_gbdt.predict( X_test) 

for day in range(7):

	result_cv['day_'+str(day)] = predcv[:,day]
	result_test['day_'+str(day)] = predtest[:,day]
	score_train.append( calsore(predtrain[:,day] , y.ix[:,day]) )
	score_gbdt.append( calsore(predcv[:,day] , y_cv.ix[:,day]) )
	score_avg.append( calsore(X_cv['avg12'] , y_cv.ix[:,day]) )
	score_opt.append( calsore(X_cv['count_'+str(7+day)] , y_cv.ix[:,day]))


	print calsore(predtrain[:,day] , y.ix[:,day])
	print calsore(predcv[:,day] , y_cv.ix[:,day])
	print calsore(X_cv['avg12'] , y_cv.ix[:,day]) 
	print calsore(X_cv['count_'+str(7+day)] , y_cv.ix[:,day]),'\n'


# #1629
# X_1629 = X[shop_id_train==1629]
# y_1629 = y[shop_id_train==1629]
# pre_1629 = bst.predict( xgb.DMatrix( X_1629) )

# temp=pd.DataFrame({'X_1629':list(X_1629['avg12']),'y_1629':list(y_1629) , 'pre_1629':pre_1629})
# print temp

result_cv.to_csv('gbdt_cv.csv',index=False)
result_test.to_csv('gbdt_test.csv',index=False,header=None)
print 'score_train:', np.mean(score_train)
print 'score_gbdt:', np.mean(score_gbdt)
print 'score_avg:', np.mean(score_avg)
print 'score_last:', np.mean(score_opt)

print 'score_gbdt:', score_gbdt
print 'score_avg:', score_avg
print 'score_last:', score_opt