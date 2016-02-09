import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #NOTE: I did not bother doing a naive version and went straight into
  #      doing a vectorized implementation
  C, D = W.shape
  D, N = X.shape #I am assuming that the W and X arguments have the same D

  F = np.dot(W, X)
  #F -= np.amax(F, axis=0) #Normalization trick to avoid numeric instability

  L = np.log(np.sum(np.exp(F), axis=0))
  L -= F[y, range(N)]

  loss = np.sum(L)/N + 0.5*reg*np.sum(W*W)
  
  Y = np.zeros( (C, N) ) 
  Y[ y, range(N)] = 1 #Matrix of 0s with one 1 per column
  dW = - np.dot(Y ,np.transpose(X))

  e_F = np.sum(np.exp(F), axis=0) #These are the denominators
  Y_prime = (np.ones((C, N))*np.exp(F)) / e_F #Like Y, Y_prime indiciates classes used
  
  dW += np.dot(Y_prime, np.transpose(X))
  dW /= N
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  #NOTE: I implemented the vectorized version in softmax_loss_naive,
  #      so I am just going to call that function
  loss, dW = softmax_loss_naive(W, X, y, reg)

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
