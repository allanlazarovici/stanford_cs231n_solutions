import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_flattened = np.reshape(x, (x.shape[0], -1))
  out = np.dot(x_flattened, w) + b
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  x_flattened = np.reshape(x, (x.shape[0], -1))

  db = np.sum(dout, axis=0)

  dw = np.dot(x_flattened.T, dout)

  dx = np.reshape(np.dot(dout, w.T), x.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.copy(x)
  out[out < 0] = 0
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  x[x<0] = 0
  x[x>0] = 1
  dx = x*dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################

  #This function is inspired by Divakar from stackoverflow.com:
  #http://stackoverflow.com/questions/30109068/
  #implement-matlabs-im2col-sliding-in-python?lq=1
  #It is similar to im2col in MATLAB, but customized for this problem
  def im2col_custom():
    #These are the indices that correspond to first "tile" in first channel
    #of first image
    start_idx = (np.arange(HH)[:,None]*W + np.arange(WW)).ravel()

    #Now we add the offsets for the channels
    start_idx = (np.arange(C)[:,None]*H*W + start_idx).ravel()[:,None]
    
    idx_1C = np.arange(H - HH + 1)[:,None][::stride]*W + np.arange(W - WW + 1)[::stride]
    idx_1C = idx_1C.ravel()

    offset_idx = (np.arange(N)[:,None]*C*H*W + idx_1C).ravel()
    return np.take(xp, start_idx + offset_idx)

  #Basic bookkeeping and formatting
  stride, pad = conv_param['stride'], conv_param['pad']
  xp = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant') 
  N, C, H, W = xp.shape
  F, _, HH, WW = w.shape
  out_height = (H - HH) / stride + 1
  out_width = (W - WW) / stride + 1

  #Get the images and weights in the right 2D matrices for multiplication
  imcols = im2col_custom() #image pixels
  w.shape = (F, C*HH*WW) #we are converting the weights into 2D

  #Matrix multiplication and re-indexing
  out = np.dot(w, imcols).T + b
  out.shape = (N, out_height*out_width, F)
  out = out.swapaxes(1,2)
  out.shape = (N, F, out_height, out_width)
  w.shape = (F, C, HH, WW) #convert the weights back into their original shape
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  #Basic bookkeeping and formatting
  x, w, b, conv_param = cache
  stride, pad = conv_param['stride'], conv_param['pad']
  xp = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant') 
  N, F, H_prime, W_prime = dout.shape
  N, C, H, W = xp.shape
  F, _, HH, WW = w.shape
  out_height = (H - HH) / stride + 1
  out_width = (W - WW) / stride + 1
  #I later realized that slicker syntax for db can be obtained using np.sum
  #db = np.sum(dout, axis=(0, 2, 3))
  dout.shape = (N*F, H_prime, W_prime)
  db = [np.sum(dout[i::F]) for i in range(F)]
  dout.shape = (N, F, H_prime, W_prime) #change to original shape

  #This function is identical to the one in conv_forward_naive 
  #Normally, I would create a new method outside the scope of both functions
  #However, I wanted to include all code in the given solution blocks, so I
  #have made an exception.
  def im2col_custom(xp, N):
    #These are the indices that correspond to first "tile" in first channel
    #of first image
    start_idx = (np.arange(HH)[:,None]*W + np.arange(WW)).ravel()

    #Now we add the offsets for the channels
    start_idx = (np.arange(C)[:,None]*H*W + start_idx).ravel()[:,None]
    
    idx_1C = np.arange(H - HH + 1)[:,None][::stride]*W + np.arange(W - WW + 1)[::stride]
    idx_1C = idx_1C.ravel()

    offset_idx = (np.arange(N)[:,None]*C*H*W + idx_1C).ravel()
    return np.take(xp, start_idx + offset_idx)

  imcols = im2col_custom(xp, N)
  dout_reshaped = dout.transpose(1,0,2,3).reshape(F, -1).T
  dw = np.dot(imcols, dout_reshaped).T.reshape(F, C, HH, WW)
  
  #We solve dx in a similar manner to dw. The only significant difference is
  #we have to construct imcols from the filter weights. 

  #I didn't realize I would be using im2col_custom() for both dw and dx, so I
  #would normally refactor the code a bit. However, at this point I am just
  #interested in moving on with learning more about CNNs.
  index_images = np.arange(C*H*W).reshape(1, C, H, W)
  imcols = np.tile(im2col_custom(index_images, 1), F)
  

  imcols_w = np.zeros((C*H*W, F*H_prime*W_prime)).T
  ws_reshaped = np.repeat(w.reshape(F,-1), H_prime*W_prime, axis=0)
  np.add.at(imcols_w, (np.arange(F*H_prime*W_prime)[:,None],imcols.T), ws_reshaped) 
  imcols_w = imcols_w.T
  
  dout_reshaped = dout.reshape(N, -1).T

  dx = np.dot(imcols_w, dout_reshaped).T.reshape(N,C,H,W)
  dx = dx[:, :, pad:-pad, pad:-pad]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################

  #Again, I've copied and pasted this function from the conv_* functions
  #Normally, I'd create a function outside of this method, but I would like
  #to keep all my code in the codeblocks given for this assignment
  def im2col_custom(xp, N):
    #These are the indices that correspond to first "tile" in first channel
    #of first image
    start_idx = (np.arange(HH)[:,None]*W + np.arange(WW)).ravel()

    #Now we add the offsets for the channels
    start_idx = (np.arange(C)[:,None]*H*W + start_idx).ravel()[:,None]
    
    idx_1C = np.arange(H - HH + 1)[:,None][::stride]*W + np.arange(W - WW + 1)[::stride]
    idx_1C = idx_1C.ravel()

    offset_idx = (np.arange(N)[:,None]*C*H*W + idx_1C).ravel()
    return np.take(xp, start_idx + offset_idx)

  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  
  N, C_orig, H, W = x.shape

  #I shouldn't have to duplicate x- the code should be more elegantly written
  #Also, using C_orig is pretty bad
  #However, I just want to move on with learning about CNNs. This ugliness
  #would be easy to fix
  x_reshaped = x.reshape(N*C_orig, 1, H, W)
  HH = pool_height
  WW = pool_width

  H_prime = 1 + (H - HH)/stride
  W_prime = 1 + (W - WW)/stride

  C = 1
  imcols = im2col_custom(x_reshaped, N*C_orig)
  out = np.amax(imcols, axis=0).reshape(N, C_orig, H_prime, W_prime)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################

  #Again, there is going to be quite a bit of copying and pasting of code from
  #conv_backward_naive/max_pool_forward_naive
  #Also, just like with max_pool_forward_naive, the code gets a bit ugly with 
  #C_orig.
  #This code is fairly easy to refactor, but I would like to move on with 
  #learning about CNNs
  def im2col_custom(xp, N):
    #These are the indices that correspond to first "tile" in first channel
    #of first image
    start_idx = (np.arange(HH)[:,None]*W + np.arange(WW)).ravel()

    #Now we add the offsets for the channels
    start_idx = (np.arange(C)[:,None]*H*W + start_idx).ravel()[:,None]
    
    idx_1C = np.arange(H - HH + 1)[:,None][::stride]*W + np.arange(W - WW + 1)[::stride]
    idx_1C = idx_1C.ravel()

    offset_idx = (np.arange(N)[:,None]*C*H*W + idx_1C).ravel()
    return np.take(xp, start_idx + offset_idx)

  x, pool_param = cache
  pool_height = pool_param['pool_height']
  pool_width = pool_param['pool_width']
  stride = pool_param['stride']
  HH = pool_height
  WW = pool_width

  N, C_orig, H, W = x.shape
  _, _, H_prime, W_prime = dout.shape

  index = np.arange(np.prod(x.shape)).reshape(N*C_orig, 1, H, W)
  #imcols_index has dimensions (N, C, H_prime, W_prime)
  C = 1
  imcols_index = im2col_custom(index, N*C_orig)

  x_reshaped = x.reshape(N*C_orig, 1, H, W)
  imcols = im2col_custom(x_reshaped, N*C_orig)
  max_pos = np.argmax(imcols, axis=0)
  max_pos = imcols_index[max_pos, np.arange(imcols_index.shape[1])]

  dout_reshaped = dout.reshape(-1)

  dx = np.zeros(np.prod(x.shape))
  np.add.at(dx, max_pos, dout_reshaped)

  dx = dx.reshape(N, C_orig, H, W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We keep each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    mask = (np.random.rand(*x.shape) < p)/p
    out = x*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################
    out = x
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache

 
def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################
    #The TODO above seems to be erroneous. I am pretty sure we need to
    #implement the backward pass, as opposed to the forward pass
    dx = dout*mask
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx
 
