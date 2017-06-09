#coding=utf-8
from __future__ import division
import pandas as pd
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


shopinfo = pd.read_csv('../data/shop_info.csv')
print shopinfo.info()

print shopinfo['cate_2_name']=='火锅'
