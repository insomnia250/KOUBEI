#coding=utf-8
import pandas as pd 
import csv

paycount = pd.read_csv('../../data/pay_count.csv')
paycount['time_stamp']=pd.to_datetime(paycount['time_stamp'])
paycount = paycount.sort_values(by='time_stamp')
paycount['weekday'] = paycount['time_stamp'].dt.dayofweek   #周一对应0
shoplist = list(paycount.shop_id.unique())


fweek={'shop_id':[],'fw0':[],'fw1':[],'fw2':[],'fw3':[],'fw4':[],'fw5':[],'fw6':[]}

for i,shop in enumerate(shoplist):
	shopdata = paycount[paycount['shop_id']==shop]

	