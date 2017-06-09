#coding=utf-8
import pandas as pd
import numpy as np

count = pd.read_csv('../../data/pay_count_cleaned3.csv')
print count
count['time_stamp'] = pd.to_datetime(count['time_stamp'])

df = count[['shop_id','time_stamp']].groupby(['shop_id']).min()
print df 
print df['time_stamp'].max()