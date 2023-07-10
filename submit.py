import numpy as np


import random

mini_batch = 1400

beta = 0.80

# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################

def grad_1(Y_trn_1,X_trn_1,p):             
  i=0
  sum_array = [0.0]*(len(X_trn_1[0]))
  while i<len(X_trn_1):
    a = Y_trn_1[i] - np.dot(p,X_trn_1[i])
    b = np.negative(X_trn_1[i])
    c = np.multiply(b,a)
    sum_array = np.add(sum_array , c)
    i=i+1
  mean=np.divide(sum_array,len(X_trn_1))
  return mean

########################################################################

def correctn(p, X_train_1, Y_train_1):                ## Implementation of correction in Projected Gradient Descent
    non_zero_index = np.argsort(-(p))[:512]
    XI = X_train_1[:, non_zero_index]
    u = np.linalg.lstsq(XI, Y_train_1, rcond=None)[0]
    updated_weights = np.zeros_like(p)
    updated_weights[non_zero_index] = u
    p = updated_weights
    return p

###################################################################

def hardthresholding(w):                    #Hardthresholding for sparsing the vector finally
    w_copy = np.copy(w)
    Kth_big = np.sort(w)[2048-512]
    i=0
    while i<len(w):
      if(w_copy[i]<Kth_big):
        w_copy[i]=0
      i=i+1
    return w_copy
 
################################################################### 

def mymodel(X_trn,Y_trn,p):
  v = np.zeros(len(p))      #Initialize v[0] for momentum method 
  iteration = 0
  while(1):
#######   
## Making the mini-batch of size 1400 and storing it in X_trn and Y_trn

     numbers = list(range(0, 1600))
     num_samples = mini_batch

     random_numbers = random.sample(numbers, num_samples)

     X_trn_1 = X_trn[random_numbers]
     Y_trn_1 = Y_trn[random_numbers]
 
 #######

 #######
   ## Calculation of the gradient  at current iteration
     grad_t = grad_1(Y_trn_1, X_trn_1, p)
     
########

#######
     w = np.add(np.multiply(beta,v) , np.multiply((1-beta),grad_t)) #momentum method to implement gradient-descent
#######

     t = np.subtract(p, w)     #update the p vector 
     t_sparse = correctn(t,X_trn_1,Y_trn_1)   #Then applying correction function

     if (iteration == 35):                                #convergence criterion
        return t_sparse
     p = t_sparse          #updating the p vector
     v = w                 #updating the momentum vector
     iteration+=1
     


def my_fit( X_trn, Y_trn ):

	p = np.linalg.lstsq(X_trn , Y_trn , rcond=None)[0]
        
	model = mymodel(X_trn,Y_trn,p)


################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# Youe method should return a 2048-dimensional vector that is 512-sparse
	# No bias term allowed -- return just a single 2048-dim vector as output
	# If the vector your return is not 512-sparse, it will be sparsified using hard-thresholding
	
	return model					# Return the trained model

