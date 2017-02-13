#-*- coding: utf-8 -*-

import numpy as np
import random
import math
from itertools import izip, count
import logging
import sparse_python
from sklearn import svm
from sklearn.metrics import accuracy_score
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# dictionary, rfSize, SKIN_DIM, M, W, trainXC_mean, trainXC_sd, 
# test시에 필요
# 이거 DB에 넣는 함수 있어야 하나???????

def extract_DB(# arg?):
	
	total_dic = dict()
	total_dic['dictionary']   = #DB에서 추출
	total_dic['rfSize']       = #DB에서 추출
	total_dic['SKIN_DIM']     = #DB에서 추출
	total_dic['M']            = #DB에서 추출
	total_dic['W']            = #DB에서 추출
	total_dic['W']            = #DB에서 추출
	total_dic['lambd']        = #DB에서 추출
	total_dic['trainXC_mean'] = #DB에서 추출
	total_dic['trainXC_sd']   = #DB에서 추출
	
	
	return total_dic

def run_test(pwd):
		
	testX = np.loadtxt('testX.txt', delimiter=',')
	testY = np.loadtxt('testY.txt', delimiter=',')
	
	testX = testX.astype('float64') 
	testY = testY.astype('int32')
	
	train_dic    = extract_DB(# arg?)
	dictionary   = train_dic['dictionary']
	rfSize       = train_dic['rfSize']
	SKIN_DIM     = train_dic['SKIN_DIM']
	M            = train_dic['M']
	W            = train_dic['W']
	lambd        = train_dic['lambd']
	trainXC_mean = total_dic['trainXC_mean'] 
	trainXC_sd   = total_dic['trainXC_sd']   
	
	#trainXC = extract_features(trainX, dictionary, rfSize, SKIN_DIM, M, W, lambd)
	testXC = sparse_python.extract_features(testX, dictionary, rfSize, SKIN_DIM, M, W, lambd)

	#tmp        = bsxfunc_row("minus", trainXC, trainXC_mean)
	#trainXCs   = bsxfunc_row("divide", tmp, trainXC_sd)
	
	tmp      =  sparse_python.bsxfunc_row("minus", testXC, trainXC_mean)
	testXCs  =  sparse_python.bsxfunc_row("divide", tmp, trainXC_sd)
	
	# def predict_y(data, labels, L):
	
	
	y_pred = predict_y(testXCs, testY, L)
	
	# 학습 정확도 확인
	accuracy_score(testY, y_pred)
	
	# 클래스를 보여줄때 y_pred를 보여주면 됨.
	
if __name__=='__main__':
	# pwd = 
	# dictionary, rfSize, SKIN_DIM, M, W, trainXC_mean, trainXC_sd 이 변수들
	# DB에서 불러올 수 있는 함수 있어야 하나?
	run_test(pwd)

