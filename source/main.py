import numpy as np
import matplotlib.pyplot as plt
import data
from auto_diff import *
from computational_nodes import *

def train(
    x: Variable,  # Input Variable node
    y: Variable,  # Target Variable čvor
    loss_node: Node, # Loss function node
    X: np.ndarray,  # Input data
    Y: np.ndarray,  # Target data
    param_niter: int = 10000,
    param_delta: float = 0.01,
    batch_size: int = None, # None means all data is used
    verbose: bool = True,
):
    def gradient_descent_step() -> float:
        # Forward pass
        loss = loss_node()
        
        # Backward pass
        _, param_grads = compute_gradients(loss_node)
        
        # Update parametara
        for param, grad in param_grads:
            param -= param_delta * grad
            
        return loss
    
    N = len(X)

    if batch_size is None:
        x.value = X
        y.value = Y

    for i in range(param_niter):
        if batch_size is None:
            loss = gradient_descent_step()
        else:
            # Stohastic gradient descent
            indices = np.random.choice(N, batch_size, replace=False)
            x.value = X[indices]
            y.value = Y[indices]
            loss = gradient_descent_step()
            
        if verbose and i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")

    if verbose:
        print(f"Final loss: {loss}")

def logreg_train(X, Y_):
    N, D = X.shape
    C = np.max(Y_) + 1
    Y = data.class_to_onehot(Y_)

    # Parameter initialization
    W = np.random.randn(C, D)
    b = np.zeros((C, 1))

    x = Variable(X, name="x")
    y = Variable(Y, name="y")

    # Mode: affine transformation + softmax
    affine = AffineTransform(x, W.T, b.T, name="affine")
    scores = Softmax(affine, name="softmax")

    # Loss function: cross entropy
    loss = SoftmaxCrossEntropyWithLogits(scores, y, name="loss")

    train(x, y, loss, X, Y, param_niter=500000, param_delta=0.1)

    return W, b

def logreg_classify(X, W, b):
    '''
    Arguments
        X:    data, np.array NxD
        W, b: logistic regression parameters, np.array CxD ; np.array Cx1

    Returns
        probs: probabilities of each class for each data point, np.array NxC
    ''' 

    x = Variable(X, name="x")
    scores = AffineTransform(x, W.T, b.T) 
    probs = Softmax(scores)()
  
    return probs

def logreg_decfun(W, b):
  def classify(X):
    return np.argmax(logreg_classify(X, W, b), axis=1)
  return classify


if __name__ == "__main__":
    np.random.seed(100)

    # get the training dataset
    X, Y_ = data.sample_gauss_2d(3, 100)

    W, b = logreg_train(X, Y_)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis=1)

    # report performance
    accuracy, recall, precision = data.eval_perf_multi(Y, Y_)
    # AP = data.eval_AP(Y_[np.max(probs, axis=1).argsort()])
    # print(accuracy, recall, precision, AP)
    print(accuracy, recall, precision, sep='\n')
    breakpoint()

    # graph the decision surface
    decfun = logreg_decfun(W, b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5)

    # graph the data points
    data.graph_data(X, Y_, Y, special=[])

    # show the plot
    plt.show()
