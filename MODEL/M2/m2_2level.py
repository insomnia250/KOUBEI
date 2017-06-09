#coding=utf-8
from __future__ import division  
import numpy as np  
from sklearn.cross_validation import KFold
from sklearn import model_selection 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier  
from sklearn.linear_model import LogisticRegression  
import pandas as pd
from pandas import DataFrame
import xgboost as xgb
from sklearn.metrics import roc_auc_score

def logloss(attempt, actual, epsilon=1.0e-15):  
    """Logloss, i.e. the score of the bioresponse competition. 
    """  
    attempt = np.clip(attempt, epsilon, 1.0-epsilon)  
    return - np.mean(actual * np.log(attempt) + (1.0 - actual) * np.log(1.0 - attempt)) 


np.random.seed(0) # seed to shuffle the train set  

# n_folds = 10  
n_folds = 5  
verbose = True  
shuffle = False  

df_train = pd.read_csv('m2_train.csv')
df_cv = pd.read_csv('m2_cv.csv')
df_test = pd.read_csv('m2_test.csv')


X_train = df_train.ix[:,8:];    print X_train.info()
y_train = df_train.ix[:,1:8];   print y_train.info()
X_cv = df_cv.ix[:,8:]  
y_cv =df_cv.ix[:,1:8]  
X_test = df_test.ix[:,1:]
if shuffle:  
    idx = np.random.permutation(y_train.size)  
    X_train = X_train[idx]  
    y_train = y_train[idx]  


skf = list(KFold(len(y_train), n_folds))  

params={
    'booster':'gbtree',
    'objective':'binary:logistic',
    'gamma':0.1,
    'max_depth':8,
    #'lambda':250,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'min_child_weight':0.6, 
    'eta': 0.04,
    'seed':69,
    'silent':1,
    'eval_metric ':'auc'
    }

num_boost_round = 100




clfsname = ['avgrate1' , 'avgrate2' , 'avgrate3' , 'avgrate4','avgrate5']  

print "Creating train and test sets for blending."  
  
dataset_blend_train = np.zeros((X_train.shape[0], len(clfsname)))  
dataset_blend_cv = np.zeros((X_cv.shape[0], len(clfsname)))  
dataset_blend_test = np.zeros((X_test.shape[0],len(clfsname)))

for j, name in enumerate(clfsname): 

    print j, name 
    dataset_blend_cv_j = np.zeros((X_cv.shape[0], len(skf)))
    dataset_blend_test_j = np.zeros((X_test.shape[0], len(skf)))  
    for i, (trainpart, apart) in enumerate(skf):  
        print "Fold", i  
        X_trainpart = X_train.ix[trainpart]  
        y_trainpart = y_train.ix[trainpart , 2+j]  
        X_apart = X_train.ix[apart]  
        y_apart = y_train.ix[apart , 2+j]

        dtrain = xgb.DMatrix( X_trainpart, y_trainpart)
        # # 交叉验证
        # bst = xgb.cv(params, dtrain, num_boost_round, nfold=5 , metrics ='auc',verbose_eval=True )
        bst = xgb.train(params, dtrain, num_boost_round)

        y_submission = bst.predict( xgb.DMatrix( X_apart) )
        temp = pd.DataFrame({'label':y_apart,'prob':y_submission})

        dataset_blend_train[apart, j] = y_submission  
        dataset_blend_cv_j[:, i] = bst.predict( xgb.DMatrix(X_cv) )
        dataset_blend_test_j[:,i] = bst.predict( xgb.DMatrix(X_test) ) # online
    dataset_blend_cv[:,j] = dataset_blend_cv_j.mean(1)
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    print("log loss : %0.8f" % (logloss(dataset_blend_cv[:,j], y_cv.ix[:,2+j])))  
    print "auc: %0.8f" % (roc_auc_score(y_cv.ix[:,2+j], dataset_blend_cv[:,j]))
print  
print "Blending."  

# params={
#     'booster':'gbtree',
#     'objective':'reg:linear',
#     'gamma':0.1,
#     'max_depth':8,
#     'lambda':250,
#     'subsample':0.7,
#     'colsample_bytree':0.3,
#     'min_child_weight':0.3, 
#     'eta': 0.04,
#     'seed':69,
#     'silent':1,
#     }
# num_boost_round2 = 250
# dtrain2 = xgb.DMatrix( dataset_blend_train, y)
# bst2 = xgb.train(params2, dtrain2, num_boost_round2)
 
# y_submission = bst2.predict( xgb.DMatrix( dataset_blend_cv ) )
# online_submission = bst2.predict( xgb.DMatrix( dataset_blend_test ) )

# print "blend result"  
# print("blending loss : %0.5f" % (calsore(y_submission , y_cv) )

