#coding=utf-8
from __future__ import division 
import pandas as pd
import numpy as np
from pandas import DataFrame
from pandas import Series
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb 

# import train data
countfeature_train = pd.read_csv('../../Feature/count_feature/train_count_feature.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_train = pd.merge(countfeature_train, shopinfofeature,how='left',on='shop_id')

shop_id_train = df_train['shop_id']
target_train = df_train.ix[:,1:15]
tgweekday_train = df_train.ix[:,15:29]
tgmonth_train = df_train.ix[:,29:43]
tgholiday_train = df_train.ix[:,43:57]
tgworkday_train = df_train.ix[:,57:71]
cmft_train = df_train.ix[:,71:]

# import cv data
countfeature_cv = pd.read_csv('../../Feature/count_feature/cv_count_feature.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_cv = pd.merge(countfeature_cv, shopinfofeature,how='left',on='shop_id')

shop_id_cv = df_cv['shop_id']
target_cv = df_cv.ix[:,1:15]
tgweekday_cv = df_cv.ix[:,15:29]
tgmonth_cv = df_cv.ix[:,29:43]
tgholiday_cv = df_cv.ix[:,43:57]
tgworkday_cv = df_cv.ix[:,57:71]
cmft_cv = df_cv.ix[:,71:]

# import test data
countfeature_test = pd.read_csv('../../Feature/count_feature/test_count_feature.csv')
shopinfofeature = pd.read_csv('../../Feature/shop_info_feature/shop_info_feature.csv')
df_test = pd.merge(countfeature_test, shopinfofeature,how='left',on='shop_id')

shop_id_test = df_test['shop_id']
tgweekday_test = df_test.ix[:,1:15]
tgmonth_test = df_test.ix[:,15:29]
tgholiday_test = df_test.ix[:,29:43]
tgworkday_test = df_test.ix[:,43:57]
cmft_test = df_test.ix[:,57:]


# train set 
cmft_train['avg'] = 0.5* cmft_train['avg1'] + 0.5*cmft_train['avg2']
target_train['avgtarget'] = target_train.mean(1)

target_train['avgrate'] = target_train['avgtarget']/cmft_train['avg']

target_train['avgrate1'] = 0
target_train.loc[target_train['avgrate']>target_train['avgrate'].quantile(0.15),'avgrate1'] = 1

target_train['avgrate2'] = 0
target_train.loc[target_train['avgrate']>target_train['avgrate'].quantile(0.30),'avgrate2'] = 1

target_train['avgrate3'] = 0
target_train.loc[target_train['avgrate']>target_train['avgrate'].quantile(0.50),'avgrate3'] = 1

target_train['avgrate4'] = 0
target_train.loc[target_train['avgrate']>target_train['avgrate'].quantile(0.70),'avgrate4'] = 1

target_train['avgrate5'] = 0
target_train.loc[target_train['avgrate']>target_train['avgrate'].quantile(0.85),'avgrate5'] = 1

countfeature_train = pd.concat([shop_id_train , 
	target_train[['avgtarget','avgrate','avgrate1','avgrate2','avgrate3','avgrate4','avgrate5']] ,
	cmft_train],axis=1)
countfeature_train.to_csv('m2_train.csv',index = False)

# cv set 
cmft_cv['avg'] = 0.5* cmft_cv['avg1'] + 0.5*cmft_cv['avg2']
target_cv['avgtarget'] = target_cv.mean(1)

target_cv['avgrate'] = target_cv['avgtarget']/cmft_cv['avg']

target_cv['avgrate1'] = 0
target_cv.loc[target_cv['avgrate']>target_cv['avgrate'].quantile(0.15),'avgrate1'] = 1

target_cv['avgrate2'] = 0
target_cv.loc[target_cv['avgrate']>target_cv['avgrate'].quantile(0.30),'avgrate2'] = 1

target_cv['avgrate3'] = 0
target_cv.loc[target_cv['avgrate']>target_cv['avgrate'].quantile(0.50),'avgrate3'] = 1

target_cv['avgrate4'] = 0
target_cv.loc[target_cv['avgrate']>target_cv['avgrate'].quantile(0.70),'avgrate4'] = 1

target_cv['avgrate5'] = 0
target_cv.loc[target_cv['avgrate']>target_cv['avgrate'].quantile(0.85),'avgrate5'] = 1

countfeature_cv = pd.concat([shop_id_cv , 
	target_cv[['avgtarget','avgrate','avgrate1','avgrate2','avgrate3','avgrate4','avgrate5']] ,
	cmft_cv],axis=1)
countfeature_cv.to_csv('m2_cv.csv',index = False)


# test set 
cmft_test['avg'] = 0.5* cmft_test['avg1'] + 0.5*cmft_test['avg2']

countfeature_test = pd.concat([shop_id_test , cmft_test],axis=1)
countfeature_test.to_csv('m2_test.csv',index = False)