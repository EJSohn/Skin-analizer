#-*- coding: utf-8 -*-

import cPickle
import numpy as np
import random
import math
from itertools import izip, count
import logging
from sklearn import svm
from sklearn.metrics import accuracy_score
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

def vec2mat_3D_reshape(vec, row, col, depth=3):
		
	if (len(vec) != depth*row*col):
		print 'error of vector length'
		exit()
	
	tmp = np.zeros((depth, row, col))
	vec_indx   = row*col
	
	indx = 0
	for i in range(depth):
		tmp[i, :, :] = np.reshape(vec[(indx*vec_indx):((indx+1)*vec_indx)], [row, col], order='F')
		indx += 1

	return tmp

def mat2vec_3D_reshape(mat3D):

	tmp             = np.zeros((mat3D.shape[0]*mat3D.shape[1]*mat3D.shape[2]))
	each_vec_length = mat3D.shape[1]*mat3D.shape[2] 
	first_vec       = np.reshape(mat3D[0, :, :], each_vec_length, order='F')
	second_vec      = np.reshape(mat3D[1, :, :], each_vec_length, order='F')
	third_vec       = np.reshape(mat3D[2, :, :], each_vec_length, order='F')

	tmp[0:each_vec_length]                       = first_vec
	tmp[each_vec_length:(2*each_vec_length)]     = second_vec
	tmp[(2*each_vec_length):(3*each_vec_length)] = third_vec
		
	return tmp

def bsxfunc_row(opt, mat, vec):

	tmp = np.zeros((mat.shape[0], mat.shape[1]))
	
	for i in range(tmp.shape[0]):
		tmp[i, :] = vec

	if   (opt == 'plus'):
		tmp = mat + tmp
	
	elif (opt == 'minus'):
		tmp = mat - tmp
	
	elif (opt == 'times'):
		tmp = mat * tmp

	elif (opt == 'divide'):
		tmp = mat / tmp

	else:
		print 'error of operator'
		exit()
	
	return tmp


def bsxfunc_col(opt, mat, vec):
	
	tmp = np.zeros((mat.shape[0], mat.shape[1]))

	for i in range(tmp.shape[1]):
		tmp[:, i] = vec

	if   (opt == 'plus'):
                tmp = mat + tmp

        elif (opt == 'minus'):
                tmp = mat - tmp

        elif (opt == 'times'):
                tmp = mat * tmp

        elif (opt == 'divide'):
                tmp = mat / tmp

        else:
                print 'error of operator'
                exit()

        return tmp

def im2col(mat, dim):

	mat = mat.T
	dim = [dim[1], dim[0]]
	
	M, N   = mat.shape
	col_ex = N - dim[1] + 1
	row_ex = M - dim[0] + 1
	
	s_idx  = np.arange(dim[0])[:, None]*N + np.arange(dim[1])
	o_idx  = np.arange(row_ex)[:, None]*N + np.arange(col_ex)
	result = np.take(mat, s_idx.ravel()[:, None] + o_idx.ravel())

	return result  
	

def ZCA_whitening(mat):

	tmp = dict()
	
	X    = mat
	cov  = np.cov(X.T)
	M    = np.mean(X, 0)
	d, V = np.linalg.eig(cov)
	D    = np.diag(1. / np.sqrt(d+0.1))
	W    = np.dot(np.dot(V, D), V.T)	
	
	X_cen = bsxfunc_row("minus", X, M)
	X_ZCA = np.dot(X_cen, W)

	tmp['X_ZCA'] = X_ZCA
	tmp['M']     = M
	tmp['W']     = W

	return tmp

def _feature_sign_checkargs(dictionary, signals, sparsity, max_iter,
                            solution):
  
    if solution is not None:
        assert solution.ndim == signals.ndim, (
            "if provided, solutions must be same number of dimensions as "
            "signals"
        )
    if signals.ndim == 1:
        assert signals.shape[0] == dictionary.shape[0], (
            "signals.ndim == 1, but signals.shape[0] !=  dictionary.shape[0]"
        )
        if solution is not None:
            assert solution.shape[0] == dictionary.shape[1], (
                "solutions array is wrong shape (ndim=1, should have first "
                "dimension %d given dictionary)" % dictionary.shape[1]
            )
    elif signals.ndim == 2:
        assert signals.shape[1] == dictionary.shape[0], (
            "signals.ndim == 2, but signals.shape[1] !=  dictionary.shape[0]"
        )
        if solution is not None:
            assert solution.shape[0] == signals.shape[0], (
                "solutions array is wrong shape (ndim=2, should have first "
                "dimension %d given signals)" % signals.shape[0]
            )
            assert solution.shape[1] == dictionary.shape[1], (
                "solutions array is wrong shape (ndim=1, should have second "
                "dimension %d given dictionary)" % dictionary.shape[1]
            )

def _feature_sign_search_single(dictionary, signal, sparsity, max_iter,
                                solution=None):
    
    sparsity = np.array(sparsity).astype(dictionary.dtype)
    effective_zero = 1e-18
   
    gram_matrix = np.dot(dictionary.T, dictionary)
    target_correlation = np.dot(dictionary.T, signal)
   
    if solution is None:
        solution = np.zeros(gram_matrix.shape[0], dtype=dictionary.dtype)
    else:
        assert solution.ndim == 1, "solution must be 1-dimensional"
        assert solution.shape[0] == dictionary.shape[1], (
            "solution.shape[0] does not match dictionary.shape[1]"
        )
       
        solution[...] = 0.
    signs = np.zeros(gram_matrix.shape[0], dtype=np.int8)
    active_set = set()
    z_opt = np.inf
         
    nz_optimal = True
    grad = - 2 * target_correlation 
    max_grad_zero = np.argmax(np.abs(grad))
    sds = np.dot(signal.T, signal)
    counter = count(0)
    while z_opt > sparsity or not nz_optimal:
        if counter.next() == max_iter:
            break
        if nz_optimal:
            candidate = np.argmax(np.abs(grad) * (signs == 0))
            log.debug("candidate feature: %d" % candidate)
            if grad[candidate] > sparsity:
                signs[candidate] = -1.
                solution[candidate] = 0.
                log.debug("added feature %d with negative sign" %
                          candidate)
                active_set.add(candidate)
            elif grad[candidate] < -sparsity:
                signs[candidate] = 1.
                solution[candidate] = 0.
                log.debug("added feature %d with positive sign" %
                          candidate)
                active_set.add(candidate)
            if len(active_set) == 0:
                break
        else:
            log.debug("Non-zero coefficient optimality not satisfied, "
                      "skipping new feature activation")
        indices = np.array(sorted(active_set))
        restr_gram = gram_matrix[np.ix_(indices, indices)]
        restr_corr = target_correlation[indices]
        restr_sign = signs[indices]
        rhs = restr_corr - sparsity * restr_sign / 2
        new_solution = np.linalg.solve(np.atleast_2d(restr_gram), rhs)
        new_signs = np.sign(new_solution)
        restr_oldsol = solution[indices]
        sign_flips = np.where(abs(new_signs - restr_sign) > 1)[0]
        if len(sign_flips) > 0:
            best_obj = np.inf
            best_curr = None
            best_curr = new_solution
            best_obj = (sds + (np.dot(new_solution,
                                      np.dot(restr_gram, new_solution))
                        - 2 * np.dot(new_solution, restr_corr))
                        + sparsity * abs(new_solution).sum())
            if log.isEnabledFor(logging.DEBUG):
               
                ocost = (sds + (np.dot(restr_oldsol,
                                       np.dot(restr_gram, restr_oldsol))
                        - 2 * np.dot(restr_oldsol, restr_corr))
                        + sparsity * abs(restr_oldsol).sum())
                cost = (sds + np.dot(new_solution,
                                     np.dot(restr_gram, new_solution))
                        - 2 * np.dot(new_solution, restr_corr)
                        + sparsity * abs(new_solution).sum())
                log.debug("Cost before linesearch (old)\t: %e" % ocost)
                log.debug("Cost before linesearch (new)\t: %e" % cost)
            else:
                ocost = None
            for idx in sign_flips:
                a = new_solution[idx]
                b = restr_oldsol[idx]
                prop = b / (b - a)
                curr = restr_oldsol - prop * (restr_oldsol - new_solution)
                cost = sds + (np.dot(curr, np.dot(restr_gram, curr))
                              - 2 * np.dot(curr, restr_corr)
                              + sparsity * abs(curr).sum())
                log.debug("Line search coefficient: %.5f cost = %e "
                          "zero-crossing coefficient's value = %e" %
                          (prop, cost, curr[idx]))
                if cost < best_obj:
                    best_obj = cost
                    best_prop = prop
                    best_curr = curr
            log.debug("Lowest cost after linesearch\t: %e" % best_obj)
            if ocost is not None:
                if ocost < best_obj and not np.allclose(ocost, best_obj):
                    log.debug("Warning: objective decreased from %e to %e" %
                              (ocost, best_obj))
        else:
            log.debug("No sign flips, not doing line search")
            best_curr = new_solution
        solution[indices] = best_curr
        zeros = indices[np.abs(solution[indices]) < effective_zero]
        solution[zeros] = 0.
        signs[indices] = np.int8(np.sign(solution[indices]))
        active_set.difference_update(zeros)
        grad = - 2 * target_correlation + 2 * np.dot(gram_matrix, solution)
        z_opt = np.max(abs(grad[signs == 0]))
        nz_opt = np.max(abs(grad[signs != 0] + sparsity * signs[signs != 0]))
        nz_optimal = np.allclose(nz_opt, 0)

    return solution, min(counter.next(), max_iter)


def feature_sign_search(dictionary, signals, sparsity, max_iter=1000,
                        solution=None):
   
    dictionary = np.asarray(dictionary)
    _feature_sign_checkargs(dictionary, signals, sparsity, max_iter, solution)
   
    signals_ndim = signals.ndim
    signals = np.atleast_2d(signals)
    if solution is None:
        solution = np.zeros((signals.shape[0], dictionary.shape[1]),
                            dtype=signals.dtype)
        orig_sol = None
    else:
        orig_sol = solution
        solution = np.atleast_2d(solution)
   
    for row, (signal, sol) in enumerate(izip(signals, solution)):
        _, iters = _feature_sign_search_single(dictionary, signal, sparsity,
                                               max_iter, sol)
        if iters >= max_iter:
            log.warning("maximum number of iterations reached when "
                        "optimizing code for training case %d; solution "
                        "may not be optimal" % iters)
    
    if orig_sol is not None and orig_sol.ndim == 1:
        solution = orig_sol
    
    elif orig_sol is None and signals_ndim == 1:
       	solution = solution.squeeze()
    
    return solution

def l2ls_learn_basis_dual(patches, S):
	
	#시간 부족 가장 간단한 약식으로..
	dictionary = np.dot(np.linalg.pinv(S), patches)

	return dictionary


def run_sc(patches, numBases, iters, lambd):
	
	dictionary = np.random.randn(numBases, patches.shape[1])
	tmp        = np.sum(np.power(dictionary, 2), 1)
	tmp        = np.sqrt(tmp + 1e-20)
	dictionary = bsxfunc_col("divide", dictionary, tmp)

	for i in range(iters):
		print 'Runing sparse coding:  iteration', i
		
		#if(i == 0):
		#	bef_S = feature_sign_search(dictionary.T, patches, lambd)
		
		#S         = feature_sign_search(dictionary.T, patches, lambd, 1000, bef_S)
		S          = feature_sign_search(dictionary.T, patches, lambd) 
		#bef_S      = S
		#dictionary = l2ls_learn_basis_dual(patches, bef_S)
		dictionary = l2ls_learn_basis_dual(patches, S)

		tmp2 = np.sum(np.power(dictionary, 2), 1)
		tmp2 = np.sqrt(tmp2 + 1e-20)
		dictionary = bsxfunc_col("divide", dictionary, tmp2) 

	return dictionary

#   trainXC = extract_features(trainX, dictionary, rfSize, SKIN_DIM, M, W, lambd)
#   def im2col(vec, dim1, dim2):
def extract_features(X, D, rfSize, SKIN_DIM, M, W, lambd):

	numBases   = D.shape[0]
	length_RGB = X.shape[1]/3
	
	XC = np.zeros((X.shape[0], numBases*2*4))
	for i in range(X.shape[0]):
		if ( i%1000 == 0):
			print 'Extracting features: ', i, ' ', X.shape[0]
		
		first_tmp  = im2col(np.reshape(X[i, 0:length_RGB], SKIN_DIM[0:2], order='F'),                  [rfSize, rfSize])
		second_tmp = im2col(np.reshape(X[i, length_RGB:(2*length_RGB)], SKIN_DIM[0:2], order='F'),     [rfSize, rfSize])
		third_tmp  = im2col(np.reshape(X[i, (2*length_RGB):(3*length_RGB)], SKIN_DIM[0:2], order='F'), [rfSize, rfSize])		

		#np.vstack 
		patches = np.vstack((first_tmp, second_tmp, third_tmp))
		patches = patches.T

		tmp1    = np.var(patches, 1) * patches.shape[1] / (patches.shape[1] - 1)
            	tmp1    = np.sqrt(tmp1 + 10)
       		tmp2    = np.mean(patches, 1)
       		tmp_mat = np.zeros((patches.shape[0], patches.shape[1]))
        	tmp_mat = bsxfunc_col("minus", patches, tmp2)
        	patches = bsxfunc_col("divide", tmp_mat, tmp1)

		patches = bsxfunc_row("minus", patches, M)
		patches = np.dot(patches, W)

		# S  = feature_sign_search(dictionary.T, patches, lambd)
		z = feature_sign_search(D.T, patches, lambd)
		patches = np.hstack((np.maximum(z, 0), -1*np.minimum(z, 0)))		

		prows  = SKIN_DIM[0] - rfSize + 1
		pcols  = SKIN_DIM[1] - rfSize + 1
		
		# def vec2mat_3D_reshape(vec, row, col, depth=3):
		patches = np.reshape(patches, patches.shape[0]*patches.shape[1], order='F')
		patches = vec2mat_3D_reshape(patches, prows, pcols, numBases*2)

		halfr  = int(round(prows/2))
		halfc  = int(round(pcols/2))

		if ((prows/2.) - halfr == 0.5):
   			halfr += 1

		if ((pcols/2.) - halfc == 0.5):
    			halfc += 1

		q1 = []
		q2 = []
		q3 = []
		q4 = []		

		for j in range(patches.shape[0]):
			q1.append(np.sum(patches[j, 0:halfr, 0:halfc]))
			q2.append(np.sum(patches[j, halfr:,  0:halfc]))
			q3.append(np.sum(patches[j, 0:halfr, halfc: ]))
			q4.append(np.sum(patches[j, halfr:,  halfc: ]))

		XC[i, :] = np.hstack((q1, q2, q3, q4))

	return XC		
	

def model_svm(data, labels, L=1.):
	#clf = svm.SVC(kernel='linear', C=L)
	clf = svm.SVC(kernel='linear')
	clf.fit(data, labels)
	y_pred = clf.predict(data)
	
	with open('/home/ljh/SC/git-repository/clf.pkl', 'wb' ) as fid:
		cPickle.dump(clf, fid)

	#tmp = dict()
	#tmp['y_pred'] = y_pred
	#tmp['model']  = clf

	return y_pred


def run():
	
	trainX = np.loadtxt('trainX.txt', delimiter=',')
	trainY = np.loadtxt('trainY.txt', delimiter=',')
	
	trainX = trainX.astype('float64')
	trainY = trainY.astype('float64')

	rfSize = 6
	numBases   = 100      #int(input("Enter the number of Dictionary basis: "))
	numPatches = 10000    # int(input("Enter the number of Patches: "))
	SKIN_DIM = [32, 32, 3]
	alpha = 0.25
	lambd = 1.0
	L = 1.0 #SVM parameter

	patches = np.zeros((numPatches, rfSize*rfSize*3))
	
	for i in range(numPatches):
			
		if ((i % 1000) == 0):
			print 'Extracting patch: ', i, '/', numPatches
		r = random.randint(0, SKIN_DIM[0] - rfSize)
		c = random.randint(0, SKIN_DIM[1] - rfSize)
		tmp = trainX[random.randint(0, (trainX.shape[0]-1)), :]
		patch = vec2mat_3D_reshape(tmp, SKIN_DIM[0], SKIN_DIM[1])
		patch = patch[:, r:r+rfSize, c:c+rfSize]
		patches[i, :] = mat2vec_3D_reshape(patch)

	tmp1    = np.var(patches, 1) * patches.shape[1] / (patches.shape[1] - 1)
	tmp1    = np.sqrt(tmp1 + 10)
	tmp2    = np.mean(patches, 1)
	tmp_mat = np.zeros((patches.shape[0], patches.shape[1]))
	tmp_mat = bsxfunc_col("minus", patches, tmp2)
	patches = bsxfunc_col("divide", tmp_mat, tmp1)
	
	ZCA_dict = dict()
	ZCA_dict = ZCA_whitening(patches)
	patches  = ZCA_dict['X_ZCA']
	M        = ZCA_dict['M']
	W        = ZCA_dict['W']
	
	dictionary = run_sc(patches, numBases, 10, lambd)

	#W, M whitening dict으로 받음
     #C    = np.dot(patches.T, patches)
     #M    = np.mean(patches, 0)
     #d, V = np.linalg.eig(C)
     #D    = np.diag(1. / np.sqrt(d+0.1))
     #W    = np.dot(np.dot(V, D), V.T)

	trainXC       = extract_features(trainX, dictionary, rfSize, SKIN_DIM, M, W, lambd)
	trainXC_mean  = np.mean(trainXC, 0)
	trainXC_sd    = np.sqrt(np.var(trainXC, 0)*trainXC.shape[1]/(trainXC.shape[1]-1)+0.01)
	
	tmp        = bsxfunc_row("minus", trainXC, trainXC_mean)
	trainXCs   = bsxfunc_row("divide", tmp, trainXC_sd)
	
	# def predict_y(data, labels, L):
	#svm_dict = dict()
	#svm_dict = model_svm(trainXCs, trainY, L)
	
	y_pred = model_svm(trainXCs, trainY, L)
	
	# 학습 정확도 확인
	print accuracy_score(trainY, y_pred)
	
	# dictionary, rfSize, SKIN_DIM, M, W, trainXC_mean, trainXC_sd, 
	# test시에 필요
	# 이거 DB에 넣는 함수 있어야 하나???????

	for_test_dict = dict()
	for_test_dict['dictionary']   = dictionary
	for_test_dict['rfSize']       = rfSize
	for_test_dict['SKIN_DIM']     = SKIN_DIN
	for_test_dict['M']            = M
	for_test_dict['W']            = W
	for_test_dict['lambd']        = lambd
	for_test_dict['trainXC_mean'] = trainXC_mean
	for_test_dict['trainXC_sd']   = trainXC_sd
	#for_test_dict['model']        = model 
	
	with open('/home/ljh/SC/git-repository/for_test_dict.pkl', 'wb') as fid:
		cPickle.dump(for_test_dict, fid)

	return for_test_dict  
	
#def DB_connect():
	
	
if __name__=='__main__':
	#pwd = 
	for_DB  = dict()
	for_DB  = run()
	#DB_connect()
	 
	

	
	











