#coding=utf-8
import pandas as pd
import os



# 商家信息
# df = pd.read_csv('../dataset/shop_info.txt',header = None,
# 	names = ['shop_id','city_name','location_id','per_pay','score',
# 	'comment_cnt','shop_level','cate_1_name','cate_2_name','cate_3_name'])
# df.to_csv('../data/shop_info.csv',index=False,coding='utf-8')


# #商家每天客流量统计
# # os.mkdir('pay_count')
# df = pd.read_csv('../dataset/user_pay.txt',header=None, 
# 	names=['user_id','shop_id','time_stamp'], chunksize = 14000000)
# for i, dfpart in enumerate(df):
# 	print i
# 	dfpart['time_stamp'] = pd.to_datetime(dfpart['time_stamp'].str[0:10])
# 	dfpart =  dfpart.groupby(['shop_id','time_stamp']).count()
# 	dfpart.columns=['count']
# 	dfpart.to_csv('./pay_count/part{0}.csv'.format(i))

#整合
# df = pd.DataFrame(columns=['shop_id','time_stamp','count'])
# for file in os.listdir('./pay_count'):
# 	dfpart = pd.read_csv('./pay_count/'+file)
# 	print file
# 	df = pd.merge(df, dfpart,on=['shop_id','time_stamp'],how='outer',suffixes=('', '_'+file[4:-4]))

# df = df.fillna(0)
# df['count'] = df.drop(['count','shop_id','time_stamp'],axis=1).sum(1)
# df = df[['shop_id','time_stamp','count']]
# print df
# df.to_csv('../data/pay_count.csv',index=False)

#商家每天浏览量统计
# os.mkdir('view_count')
# df = pd.read_csv('../dataset/user_view.txt',header=None, 
# 	names=['user_id','shop_id','time_stamp'])
# df['time_stamp'] = pd.to_datetime(df['time_stamp'].str[0:10])
# df =  df.groupby(['shop_id','time_stamp']).count()
# df.columns=['count']
# df.to_csv('./view_count/part{0}.csv'.format(1))

# df = pd.read_csv('../dataset/extra_user_view.txt',header=None, 
# 	names=['user_id','shop_id','time_stamp'])
# df['time_stamp'] = pd.to_datetime(df['time_stamp'].str[0:10])
# df =  df.groupby(['shop_id','time_stamp']).count()
# df.columns=['count']
# df.to_csv('./view_count/part{0}.csv'.format(2))

# #整合
# df = pd.DataFrame(columns=['shop_id','time_stamp','count'])
# for file in os.listdir('./view_count'):
# 	dfpart = pd.read_csv('./view_count/'+file)
# 	print file
# 	df = pd.merge(df, dfpart,on=['shop_id','time_stamp'],how='outer',suffixes=('', '_'+file[4:-4]))

# df = df.fillna(0)
# df['count'] = df.drop(['count','shop_id','time_stamp'],axis=1).sum(1)
# df = df[['shop_id','time_stamp','count']]
# print df
# df.to_csv('../data/view_count.csv',index=False)


'''天气'''
df = pd.read_csv('../dataset/city_weather/ankang',header=None,names=['time_stamp','t1','t2','weather','wind','windlevel'])
print df