#-*- coding: utf-8 -*-

# 1레벨

import cPickle
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

def run_test():
		
	#DB에서 data뽑기 
	testX = np.loadtxt('testX.txt', delimiter=',')  ############### input에 맞게 고치기
	testY = np.loadtxt('testY.txt', delimiter=',')
	
	testX = testX.astype('float64')
	testY = testY.astype('float64')

	testX = testX[5, :]
	testY = testY[5]

	train_dic = dict()
	with open('/home/ljh/SC/git-repository/for_test_dict.pkl', 'rb') as fid:
		train_dic = cPickle.load(fid)

	with open('/home/ljh/SC/git-repository/clf.pkl', 'rb' ) as fid:
		clf_loaded = cPickle.load(fid)

	dictionary   = train_dic['dictionary']
	rfSize       = train_dic['rfSize']
	SKIN_DIM     = train_dic['SKIN_DIM']
	M             = train_dic['M']
	W             = train_dic['W']
	lambd         = train_dic['lambd']
	trainXC_mean = total_dic['trainXC_mean'] 
	trainXC_sd    = total_dic['trainXC_sd']
	#model         = total_dic['model']   
	
	#trainXC = extract_features(trainX, dictionary, rfSize, SKIN_DIM, M, W, lambd)
	testXC = sparse_python.extract_features(testX, dictionary, rfSize, SKIN_DIM, M, W, lambd)

	#tmp        = bsxfunc_row("minus", trainXC, trainXC_mean)
	#trainXCs   = bsxfunc_row("divide", tmp, trainXC_sd)
	
	tmp      =  sparse_python.bsxfunc_row("minus", testXC, trainXC_mean)
	testXCs  =  sparse_python.bsxfunc_row("divide", tmp, trainXC_sd)
	
	# def predict_y(data, labels, L):
	test_y_pred = clf_loaded.predict(testXCs) #################### <--- output
	
	# 학습 정확도 확인
	print accuracy_score(testY, test_y_pred)
	return test_y_pred
	# 클래스를 보여줄때 y_pred를 보여주면 됨.
	
if __name__=='__main__':
	# pwd = 
	# dictionary, rfSize, SKIN_DIM, M, W, trainXC_mean, trainXC_sd 이 변수들
	# DB에서 불러올 수 있는 함수 있어야 하나?
	run_test()

