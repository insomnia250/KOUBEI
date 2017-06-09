#coding=utf-8
from __future__ import division 
import numpy as np
from pandas import DataFrame
import pandas as pd

def calsore(pred,target):
	temp = 1.0 * np.abs(pred - target) / np.abs(pred + target)
	return temp.mean()

def obj_function(preds, dtrain):
	labels = dtrain.get_label()
	sgn = np.sign(preds-labels)

	grad = sgn * 1e16 * labels / (preds+labels)**2
	hess = -sgn * 1e16 * labels / (preds+labels)**3
	# print grad
	# print hess
	return grad, hess

def obj_function3(preds, dtrain):
	labels = dtrain.get_label()

	grad = 4.0 * (preds-labels)/(preds+labels) * labels/(preds+labels)**2
	hess = 8.0 * labels * (2*labels - preds)/(preds+labels)**4
	# print grad
	# print hess
	return grad*10e5, hess*10e5


def eva_function(preds, dtrain):
	labels = dtrain.get_label()
	temp = 1.0 * np.abs(preds - labels) / np.abs(preds + labels)
	return 'MAPE1', temp.mean()

def print_ft_impts(featureColumns,bst):
	# 打印特征重要性
	FeatureImportance = DataFrame(bst.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)
	list1 = []
	for fNum in range(len(featureColumns)):
		list1.append('f'+str(fNum))
	print FeatureImportance
# def obj_function2(preds, dtrain):
# 	labels = dtrain.get_label()
# 	sgn = np.sign(preds-labels)

# 	grad = 2* (preds-labels)
# 	hess = 2*np.ones(len(grad))
# 	print grad
# 	print hess
# 	return grad, hess


def skobj3(y_true, y_pred):
	c = y_true
	y = y_pred
	epsilon = 1e-5
	grad = -((2*c - 2*y)/(c + y)**2 + (2*(c - y)**2)/(c + y)**3)/(2*(epsilon + (c - y)**2/(c + y)**2)**(1/2))
	hess = (2/(c + y)**2 + (4*(2*c - 2*y))/(c + y)**3 + (6*(c - y)**2)/(c + y)**4)/(2*(epsilon + (c - y)**2/(c + y)**2)**(1/2)) - ((2*c - 2*y)/(c + y)**2 + (2*(c - y)**2)/(c + y)**3)**2/(4*(epsilon + (c - y)**2/(c + y)**2)**(3/2))
	grad = grad
	hess = hess
	# grad = 2* (preds-labels)
	# hess = 2*np.ones(len(grad))
	# print grad
	# print 'aaa',hess
	return grad, hess