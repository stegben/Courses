import numpy as np
import theano
import theano.tensor as T

from MLP import MLP

"""
Train a fully connected multilayer NNet
Tradition Back prop, without pre-training
with: 
	1. L1 and L2 regularization
	2. adjustable mini-batch SGD
	3. momentum
"""


"""
TODO: 
1. momentumGradientUpdate
2. 
"""
def MLPtrainer(net ,
	           learning_rate = 0.1,
  	           momentum = 0.9 ,
  	           L1 = 0.0 ,
  	           L2 = 0.0 ):

  input = T.matrix('input') # shape(n_instances , n_features)
  label = T.matrix('label') # shape(n_instances , n_labels)
  
  # cost = net.squareError(x=input,y=label)
  # cost = net.crossEntropyError(x=input , y=label)
  cost = costFntWithL1L2(net,L1,L2,input,label)

  # updates = fixedLearningRateUpdate(cost , net.params , learning_rate)

  updates = momentumGradientUpdate(cost , net.params , learning_rate , momentum)
  
  train_model = theano.function(
             inputs = [input,label],
             outputs = cost ,
             updates = updates
  	)
  return train_model
  

def costFntWithL1L2(net , L1 , L2 ,x,y):
  regL1 = 0
  for layer in net.layers:
  	regL1 += abs(layer.W).mean()

  regL2 = 0
  for layer in net.layers:
  	regL2 += (layer.W ** 2).mean()

  cost = (
          net.rationalError(x,y)
        + L1 * regL1
        + L2 * regL2
    )
  return cost

def fixedLearningRateUpdate(cost , params , learning_rate):
  gparams = [T.grad(cost, param) for param in params]

  updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(params, gparams)
  ]
  return updates




def momentumGradientUpdate(cost, params, learning_rate, momentum):
  '''
  Compute updates for gradient descent with momentum
    
  :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
  :returns:
        updates : list
            List of updates, one for each parameter
  '''
  # Make sure momentum is a sane value
  assert momentum < 1 and momentum >= 0
  # List of update steps for each parameter
  updates = []
  # Just gradient descent on cost
  for param in params:
  	# For each parameter, we'll create a param_update shared variable.
    # This variable will keep track of the parameter's update step across iterations.
    # We initialize it to 0
    param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
    # Each parameter is updated by taking a step in the direction of the gradient.
    # However, we also "mix in" the previous step according to the given momentum value.
    # Note that when updating param_update, we are using its old value and also the new gradient step.
    updates.append((param, param - learning_rate*param_update))
    # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
    updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
  return updates


# def updateModel(net,x,y):