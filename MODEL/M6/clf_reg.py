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
import cPickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

#################################ＣＬＦ　　label: target/avg12 

# import train1 data  training CLFs
countfeature_train1 = pd.read_csv('../../Feature/count_feature/cl2/train_countcl_featureTG7.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_train1 = pd.merge(countfeature_train1, shopinfofeature,how='left',on='shop_id')


for i in range(7):
	df_train1['fw'+str(i)+'avg1'] = df_train1['fw'+str(i)]*df_train1['avg1']
	df_train1['fw'+str(i)+'avg12'] = df_train1['fw'+str(i)]*df_train1['avg12']

shop_id_train1 = df_train1['shop_id']
target_train1 = df_train1.ix[:,1:8]
tgweekday_train1 = df_train1.ix[:,8:15]
tgmonth_train1 = df_train1.ix[:,15:22]
tgworkday_train1 = df_train1.ix[:,22:29]
dayopened_train1 = df_train1.ix[:,29:36]
cmft_train1 = df_train1.ix[:,36:]


# import train2 data   training REG
countfeature_train2 = pd.read_csv('../../Feature/count_feature/cl2/cv_countcl_featureTG7.csv');  
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_train2 = pd.merge(countfeature_train2, shopinfofeature,how='left',on='shop_id')
# for i in range(13):
# 	df_train2.drop(['count_'+str(i)],axis=1,inplace=True)
for i in range(7):
	df_train2['fw'+str(i)+'avg1'] = df_train2['fw'+str(i)]*df_train2['avg1']
	df_train2['fw'+str(i)+'avg12'] = df_train2['fw'+str(i)]*df_train2['avg12']

shop_id_train2 = df_train2['shop_id']
target_train2 = df_train2.ix[:,1:8]
tgweekday_train2 = df_train2.ix[:,8:15]
tgmonth_train2 = df_train2.ix[:,15:22]
tgworkday_train2 = df_train2.ix[:,22:29]
dayopened_train2 = df_train2.ix[:,29:36]
cmft_train2 = df_train2.ix[:,36:]

# import cv data   offline test
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




# training
score_xgb = []; score_avg = [];  score_opt = [];  score_train=[]
result_cv = DataFrame(); result_cv['shop_id'] = shop_id_cv
# result_test = DataFrame(); result_test['shop_id'] = shop_id_test
for day in range(7):
	# if day==1:continue
	aucs = []
	X1 = pd.concat([tgweekday_train1.ix[:,day] , tgmonth_train1.ix[:,day] ,
		tgworkday_train1.ix[:,day] , cmft_train1],axis=1)
	X1.drop(['tgmonth_'+str(day)],axis=1,inplace=True)
	
	# 比值的分位数做标签
	X1['rate'] = target_train1.ix[:,day]/X1['avg12']
	X1['clf1_label'] = 0
	X1.loc[X1['rate']>X1['rate'].quantile(0.12),'clf1_label'] = 1

	X1['clf2_label'] = 0
	X1.loc[X1['rate']>X1['rate'].quantile(0.25),'clf2_label'] = 1	

	X1['clf3_label'] = 0
	X1.loc[X1['rate']>X1['rate'].quantile(0.4),'clf3_label'] = 1	
	
	X1['clf4_label'] = 0
	X1.loc[X1['rate']>X1['rate'].quantile(0.5),'clf4_label'] = 1		

	X1['clf5_label'] = 0
	X1.loc[X1['rate']>X1['rate'].quantile(0.6),'clf5_label'] = 1	

	X1['clf6_label'] = 0
	X1.loc[X1['rate']>X1['rate'].quantile(0.75),'clf6_label'] = 1

	X1['clf7_label'] = 0
	X1.loc[X1['rate']>X1['rate'].quantile(0.88),'clf7_label'] = 1

	clf_labels = X1[['clf1_label','clf2_label','clf3_label','clf4_label','clf5_label','clf6_label','clf7_label']]
	X1.drop(['rate','clf1_label','clf2_label','clf3_label','clf4_label','clf5_label','clf6_label','clf7_label'],axis=1,inplace=True)

	# 训练这些预测分位数的分类器
	clfs = []
	for i in range(len(clf_labels.columns)):
		clf_y = clf_labels.ix[:,i]
		params={
		'booster':'gbtree',
		'objective':'binary:logistic',
		'eval_metric ':'auc',
		'gamma':0.1,
		'max_depth':8,
		# 'lambda':10,
		'subsample':0.7,
		'colsample_bytree':0.3,
		'min_child_weight':0.3, 
		'eta': 0.04,
		'seed':69,
		'silent':1,}
		dtrain1 = xgb.DMatrix( X1, clf_y)

		# # 交叉验证,obj=obj_function
		# clf = xgb.cv(params, dtrain1, num_boost_round=500 , metrics ='auc', nfold=5,verbose_eval=True)
		# #训练
		clf = xgb.train(params, dtrain1, num_boost_round=30)#,evals=watchlist)
		# bst.save_model('./clf_models/day{0}_clf{1}.model'.format(day,i))
		clfs.append(clf)
	#########################################################
	# train2集，训练REG
	X2 = pd.concat([tgweekday_train2.ix[:,day] , tgmonth_train2.ix[:,day] ,
		tgworkday_train2.ix[:,day] , cmft_train2],axis=1)
	X2.drop(['tgmonth_'+str(day)],axis=1,inplace=True)

	X2['rate'] = target_train2.ix[:,day]/X2['avg12']
	X2['clf1_label'] = 0
	X2.loc[X2['rate']>X2['rate'].quantile(0.12),'clf1_label'] = 1

	X2['clf2_label'] = 0
	X2.loc[X2['rate']>X2['rate'].quantile(0.25),'clf2_label'] = 1	

	X2['clf3_label'] = 0
	X2.loc[X2['rate']>X2['rate'].quantile(0.4),'clf3_label'] = 1	
	
	X2['clf4_label'] = 0
	X2.loc[X2['rate']>X2['rate'].quantile(0.5),'clf4_label'] = 1		

	X2['clf5_label'] = 0
	X2.loc[X2['rate']>X2['rate'].quantile(0.6),'clf5_label'] = 1	

	X2['clf6_label'] = 0
	X2.loc[X2['rate']>X2['rate'].quantile(0.75),'clf6_label'] = 1

	X2['clf7_label'] = 0
	X2.loc[X2['rate']>X2['rate'].quantile(0.88),'clf7_label'] = 1

	clf_labels = X2[['clf1_label','clf2_label','clf3_label','clf4_label','clf5_label','clf6_label','clf7_label']]
	X2.drop(['rate','clf1_label','clf2_label','clf3_label','clf4_label','clf5_label','clf6_label','clf7_label'],axis=1,inplace=True)
	reg_y = target_train2.ix[:,day]

	level2_ft = DataFrame()
	for i in range(len(clf_labels.columns)):
		clf = clfs[i]
		pred_train2 = clf.predict( xgb.DMatrix( X2) )
		level2_ft['clf'+str(i)] = pred_train2
		auc = roc_auc_score(clf_labels.ix[:,i], pred_train2)
		aucs.append(auc)
	# print aucs
	level2_ft=pd.concat([level2_ft,X2],axis=1)
	print 'day{0},mean auc:{1}'.format(day,np.mean(aucs))
	# print level2_ft.info()

	params2={
    'booster':'gbtree',
    'objective':'reg:linear',
    'gamma':0.1,
    'max_depth':8,
    # 'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.3,
    'min_child_weight':0.3, 
    'eta': 0.04,
    'seed':69,
    'silent':1,
    }
	dtrain = xgb.DMatrix( level2_ft, reg_y)


	reg = xgb.train(params2, dtrain, num_boost_round=130)
	predtrain = reg.predict( xgb.DMatrix( level2_ft) )
	# # # 打印特征重要性
	# featureColumns = level2_ft.columns
	# print_ft_impts(featureColumns,reg)
	##################################################################
	#线下测试2level
	X_cv = pd.concat([tgweekday_cv.ix[:,day] , tgmonth_cv.ix[:,day] ,
		tgworkday_cv.ix[:,day] , cmft_cv],axis=1)
	X_cv.drop(['tgmonth_'+str(day)],axis=1,inplace=True)
	y_cv = target_cv.ix[:,day]

	#level1 CLFs
	level2_ft_cv = DataFrame()
	for j in range(len(clfs)):
		level2_ft_cv['clf'+str(j)] = clfs[j].predict( xgb.DMatrix( X_cv) )

	level2_ft_cv=pd.concat([level2_ft_cv,X_cv],axis=1)

	predcv = reg.predict( xgb.DMatrix( level2_ft_cv) )
	result_cv['day_'+str(day)] = list(predcv)

	score_train.append( calsore(predtrain , reg_y) )
	score_xgb.append( calsore(predcv , y_cv) )
	print calsore(predtrain , reg_y)
	print calsore(predcv , y_cv)

print 'score_train:', np.mean(score_train)
print 'score_xgb:', np.mean(score_xgb)


print 'score_xgb:', score_xgb
