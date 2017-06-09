#coding=utf-8
# from __future__ import print_function
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot

paycount = pd.read_csv('../data/pay_count.csv')
# paycount['count'] = np.log1p(paycount['count'])
paycount['time_stamp'] = pd.to_datetime(paycount['time_stamp'])
shopdata = paycount[paycount['shop_id']==10]

startday = shopdata['time_stamp'].min()
endday = shopdata['time_stamp'].max()


# dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422, 
# 6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355, 
# 10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767, 
# 12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232, 
# 13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248, 
# 9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722, 
# 11999,9390,13481,14795,15845,15271,14686,11054,10395];

# dta=pd.Series(dta); dta = dta.astype(float)
# dta.index = pd.Index(pd.date_range(start=pd.datetime(2016,10,1),periods=90,freq='D' ))
# print dta.index 
dta = shopdata['count']; dta.index = shopdata['time_stamp']


oridta = dta
# diffdta = dta[1:]
diffdta= dta.diff(1)

plt.plot(diffdta.index,diffdta,marker='o')
plt.show()

from statsmodels.tsa.stattools import adfuller
dftest = adfuller(diffdta.iloc[1:], autolag='AIC')
print 'p-value:%.10f' %dftest[1]


# # grid search for ARMA paras
# res = sm.tsa.stattools.arma_order_select_ic(diffdta.iloc[1:], max_ar=7, max_ma=7, ic='aic', trend='nc')
# print res.aic_min_order

# fitting using minAIC paras
arma_mod = sm.tsa.ARMA(diffdta.iloc[1:],res.aic_min_order).fit(disp=-1)
print(arma_mod.aic,arma_mod.bic,arma_mod.hqic)

# print(sm.stats.durbin_watson(arma_mod70.resid.values))
# print(sm.stats.durbin_watson(arma_mod66.resid.values))



predict_diff = arma_mod.predict(endday+pd.to_timedelta('1day'), '2016-11-14', dynamic=True)
# predict_diff = arma_mod.predict(pd.datetime(2016,12,30), pd.datetime(2017,1,15), dynamic=True)


plt.figure()
plt.plot(arma_mod.fittedvalues,'r')
plt.plot(diffdta,'b')


diffdta.iloc[0] = oridta.iloc[0]
plt.figure()
plt.plot(arma_mod.fittedvalues.cumsum() + oridta.iloc[0],'r')
plt.plot(oridta.index,oridta,'b')
plt.plot(diffdta.cumsum(),'k')
plt.plot(predict_diff.index,predict_diff.cumsum() + oridta.iloc[-1] , 'g')


plt.show()


# 3.0
# 4.0
# 8.0
# 10.0
# 15.0
# 20.0
# 30.0
# 31.0
# 33.0
# 40.0
# 43.0