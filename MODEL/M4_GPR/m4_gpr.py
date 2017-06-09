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
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
import cPickle

paycount = pd.read_csv('../../data/pay_count_cleaned3.csv')
paycount['time_stamp']=pd.to_datetime(paycount['time_stamp'])
paycount = paycount.sort_values(by=['time_stamp'])
paycount['weekday'] = paycount['time_stamp'].dt.dayofweek 

shoplist = list( paycount['shop_id'].unique() )

# training
score_gp = []; score_avg = [];  score_opt = [];  score_train=[]
result_cv = DataFrame();
result_test = DataFrame(); 
for day in range(1):
	# if day==1:break
	y_cv=[];predcv=[]
	resultshoplist=[]
	for i,shop in enumerate(shoplist):
		print i,shop
		# [168]
		# if i <168:continue
		weekday = (pd.datetime(2016,11,1)+pd.to_timedelta(str(day)+'days')).weekday() 
		shopdata = paycount[(paycount['shop_id']==shop) & (paycount['weekday']==weekday)]

		X = np.arange(len(shopdata)).reshape(-1,1)[0:-1]
		y = shopdata['count'].values[0:-1]

		X_cv = np.array([len(X)]).reshape(-1,1)

		X_test = np.array([len(X)+1+day]).reshape(-1,1)

		'''
		gp
		'''
		# Kernel with optimized parameters
		k1 = 50.0**2 * RBF(length_scale=50.0)  # long term smooth rising trend
		k2 = 2.0**2 * RBF(length_scale=100.0) \
		    * ExpSineSquared(length_scale=1.0, periodicity=1.0,
		    periodicity_bounds="fixed")  # seasonal component
		# medium term irregularities
		k3 = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
		k4 = 0.1**2 * RBF(length_scale=0.1) \
		+ WhiteKernel(noise_level=0.1**2,
		noise_level_bounds=(1e-3, np.inf))  # noise terms
		kernel =k1+ k2 + k3 +k4

		gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
			normalize_y=True)

		try:
			gp.fit(X,y)
		except:
			continue
		else:
			#cv
			resultshoplist.append(shop)
			y_cv.append(shopdata['count'].values[-1])
			predcv.append(gp.predict(X_cv)[0])


			# #plot
			# X_ = np.linspace(X.min(), X.max() + 30, len(X+30) )[:, np.newaxis]

			# y_pred, y_std = gp.predict(X_, return_std=True)

			# # Illustration
			# plt.scatter(X, y, c='k')
			# plt.plot(X_, y_pred)

			# plt.plot(X_, y_pred - y_std,'y')
			# plt.plot(X_, y_pred + y_std,'y')
			# plt.xlim(X_.min(), X_.max())
			# plt.xlabel("Year")
			# plt.ylabel(r"CO$_2$ in ppm")
			# plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
			# plt.tight_layout()
			# plt.show()


	result_cv['day_'+str(day)] = predcv

	error = np.abs(result_cv['day_'+str(day)]-y_cv)/(result_cv['day_'+str(day)]+y_cv)
	ers = pd.DataFrame({'shop_id':resultshoplist,'error':error})

	print ers.sort_values(by='error')

	score_gp.append( calsore(np.array(predcv) ,np.array( y_cv)) )


	print calsore(np.array(predcv) , np.array(y_cv))

# 	print calsore(X_cv['avg12'] , y_cv) 
# 	print calsore(target_cv.mean(1) , y_cv),'\n'
# 	#test
# 	predtest = gp.predict( X_test) 
# 	result_test['day_'+str(day)] = list(predtest)

# 	# #1629
# 	# X_1629 = X[shop_id_train==1629]
# 	# y_1629 = y[shop_id_train==1629]
# 	# pre_1629 = bst.predict( xgb.DMatrix( X_1629) )

# 	# temp=pd.DataFrame({'X_1629':list(X_1629['avg12']),'y_1629':list(y_1629) , 'pre_1629':pre_1629})
# 	# print temp
result_cv['shop_id'] = resultshoplist
result_test['shop_id'] = resultshoplist
# result_cv.to_csv('gp_cv.csv',index=False)
# result_test.to_csv('gp_test.csv',index=False,header=None)
# print 'score_train:', np.mean(score_train)
# print 'score_gp:', np.mean(score_gp)
# print 'score_avg:', np.mean(score_avg)
# print 'score_opt:', np.mean(score_opt)

# print 'score_gp:', score_gp
# print 'score_avg:', score_avg
# print 'score_opt:', score_opt