import numpy as np
from computational_nodes import *
from auto_diff import *
from copy import deepcopy
import time
from model import LogisticRegression
import data
import matplotlib.pyplot as plt


def test_add():
    print("Testing Add")

    x = Variable(2.0)
    y = Variable(3.0)
    output = Add(x, y)

    expected_forward = np.array([[5]])
    expected_wrt_x = np.array([[1]])
    expected_wrt_y = np.array([[1]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "Add forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_wrt_x), "Add grad w.r.t x failed"

    # print(grads[y])
    assert np.allclose(grads[y], expected_wrt_y), "Add grad w.r.t y failed"

    print("Add successful\n")


def test_sub():
    print("Testing Sub")

    x = Variable(5.0)
    y = Variable(3.0)
    output = Sub(x, y)

    expected_forward = np.array([[2]])
    expected_grad_wrt_x = np.array([[1]])
    expected_grad_wrt_y = np.array([[-1]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "Sub forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "Sub grad w.r.t x failed"

    # print(grads[y])
    assert np.allclose(grads[y], expected_grad_wrt_y), "Sub grad w.r.t y failed"

    print("Sub successful\n")


def test_mul():
    print("Testing Mul")

    x = Variable(4.0)
    y = Variable(5.0)
    output = Mul(x, y)

    expected_forward = output()
    expected_grad_wrt_x = np.array([[5]])
    expected_grad_wrt_y = np.array([[4]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "Mul forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "Mul grad w.r.t x failed"

    # print(grads[y])
    assert np.allclose(grads[y], expected_grad_wrt_y), "Mul grad w.r.t y failed"

    print("Mul successful\n")


def test_div():
    print("Testing Div")

    x = Variable(8.0)
    y = Variable(2.0)
    output = Div(x, y)

    expected_forward = np.array([[4]])
    expected_grad_wrt_x = np.array([[0.5]])
    expected_grad_wrt_y = np.array([[-2]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "Div forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "Div grad w.r.t x failed"

    # print(grads[y])
    assert np.allclose(grads[y], expected_grad_wrt_y), "Div grad w.r.t y failed"

    print("Div successful\n")


def test_basic_arithmetic_expression():
    print("Testing basic arithmetic expression")

    x = Variable(2.0)
    y = Variable(3.0)
    add_node = Add(x, y)
    sub_node = Sub(x, y)
    output = Mul(add_node, sub_node)

    expected_forward = np.array([[-5]])
    expected_grad_wrt_x = np.array([[4]])
    expected_grad_wrt_y = np.array([[-6]])

    # Forward
    # print(complex_node())
    assert np.allclose(
        output(), expected_forward
    ), "Basic arithmetic expression forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(
        grads[x], expected_grad_wrt_x
    ), "Basic arithmetic expression grad w.r.t x failed"

    # print(grads[y])
    assert np.allclose(
        grads[y], expected_grad_wrt_y
    ), "Basic arithmetic expression grad w.r.t y failed"

    print("Basic arithmetic expression successful\n")


def test_exp():
    print("Testing Exp")

    x = Variable(2.0)
    output = Exp(x)

    expected_forward = [[np.exp(2)]]
    expected_grad_wrt_x = [[np.exp(2)]]

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "Exp forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "Exp grad w.r.t x failed"

    print("Exp successful\n")


def test_log():
    print("Testing Log")

    x = Variable(2.0)
    output = Log(x)

    expected_forward = np.array([[np.log(2)]])
    expected_grad_wrt_x = np.array([[0.5]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "Logarithm failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "Log grad w.r.t x failed"

    print("Log successful\n")


def test_powconst():
    print("Testing PowConst")

    x = Variable(3.0)
    output = PowConst(x, 2)

    expected_forward = np.array([[9]])
    expected_grad_wrt_x = np.array([[6]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "PowConst forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "PowConst grad w.r.t x failed"

    print("PowConst successful\n")


def test_affinetransform():
    print("Testing AffineTransform")

    x = Variable(np.array([[1, 2, 3]]))
    weights = Parameter(shape=(2, 3))
    weights.value = np.array([[2, 1, 3], [1, 2, 3]])
    bias = Parameter(shape=(2, 1))
    bias.value = np.array([[1], [1]])
    
    output = AffineTransform(x, weights, bias)

    expected_forward = np.array([[14, 15]])
    expected_grad_wrt_x = np.array([[3, 3, 6]])
    expected_grad_wrt_weights = np.array([[1, 2, 3], [1, 2, 3]])
    expected_grad_wrt_bias = np.array([[1], [1]])

    # Forward
    assert np.allclose(output(), expected_forward), "AffineTransform forward failed"

    # Backward
    input_grads, param_grads = compute_gradients(output)

    assert np.allclose(
        input_grads[x], expected_grad_wrt_x
    ), "AffineTransform input grad w.r.t x failed"

    # Gradient w.r.t to weights
    assert np.allclose(
        param_grads[0][1], expected_grad_wrt_weights
    ), "AffineTransform param grad w.r.t weights failed"

    # Gradient w.r.t to bias
    assert np.allclose(
        param_grads[1][1], expected_grad_wrt_bias
    ), "AffineTransform param grad w.r.t bias failed"

    print("AffineTransform successful\n")


def test_matmul():
    print("Testing MatMul")

    A = Variable(np.array([[1, 2], [3, 4]]))
    B = Variable(np.array([[5, 6], [7, 8]]))
    output = MatMul(A, B)

    expected_forward = np.array([[19, 22], [43, 50]])
    expected_grad_wrt_a = np.array([[11, 15], [11, 15]])
    expected_grad_wrt_b = np.array([[4, 4], [6, 6]])

    # Forward
    # print(matmul_node())
    assert np.allclose(output(), expected_forward), "MatMul forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # Gradient w.r.t to A
    # print(grads[A])
    assert np.allclose(grads[A], expected_grad_wrt_a), "MatMul grad w.r.t A failed"

    # Gradient w.r.t to B
    # print(grads[B])
    assert np.allclose(grads[B], expected_grad_wrt_b), "MatMul grad w.r.t B failed"

    print("MatMul successful\n")


def test_sigmoid():
    print("Testing Sigmoid")

    x = Variable(0.0)
    output = Sigmoid(x)

    expected_forward = np.array([[0.5]])
    expected_grad_wrt_x = np.array([[0.25]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "Sigmoid forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "Sigmoid grad w.r.t x failed"

    print("Sigmoid successful\n")


def test_relu():
    print("Testing ReLU")

    x = Variable(np.array([-1, 0, 1]))
    output = ReLU(x)

    expected_forward = np.array([[0], [0], [1]])
    expected_grad_wrt_x = np.array([[0], [0], [1]])

    # Forward
    # print(z())
    assert np.allclose(output(), expected_forward), "ReLU forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "ReLU grad w.r.t x failed"

    print("ReLU successful\n")


def test_softmax():
    print("Testing Softmax")

    x = Variable(np.array([[1.0, 2.0, 3.0]]))
    output = Softmax(x)

    expected_forward = np.exp(np.array([[1.0, 2.0, 3.0]])) / np.sum(
        np.exp(np.array([[1.0, 2.0, 3.0]]))
    )
    expected_grad_wrt_x = expected_forward * (
        1 - np.sum(expected_forward, axis=-1, keepdims=True)
    )

    # Forward
    # print(softmax_node())
    assert np.allclose(output(), expected_forward), "Softmax forward failed"

    # Backward
    grads, _ = compute_gradients(output)

    # print(grads[x])
    assert np.allclose(grads[x], expected_grad_wrt_x), "Softmax grad w.r.t x failed"

    print("Softmax successful\n")


def test_gradient_descent_input():
    print("Testing gradient descent for input variable")

    x = Variable(5.0)
    output = PowConst(x, 2)

    optimized_value = gradient_descent_input(output, x, 5.0)
    # print(optimized_value)  # Expected: Close to 0.0
    assert abs(optimized_value - 0) < 1e-5

    print("Gradient descent for input variable successful\n")


def test_gradient_descent_params():
    print("Testing gradient descent for params")

    # Linear regression optimization
    # Generate some simple linear data
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    true_w, true_b = 2.5, -1.0
    y = true_w * X + true_b + np.random.randn(100, 1) * 0.1

    # Define the model
    X_var = Variable(X)
    y_var = Variable(y)
    
    # Promjena u Parameter objekte
    W = Parameter(shape=(1, 1))
    W.value = np.random.randn(1, 1) * 0.1
    
    b = Parameter(shape=(1, 1))
    b.value = np.zeros((1, 1))

    # Linear model: y = Wx + b
    affine = AffineTransform(X_var, W, b)

    # Mean squared error loss
    diff = Sub(affine, y_var)
    loss = PowConst(diff, 2)

    # Train
    gradient_descent_params(loss, param_niter=1000, param_delta=0.0001)

    # Check if parameters are close to true values
    learned_w = W.value.flatten()[0]
    learned_b = b.value.flatten()[0]

    assert (
        np.abs(learned_w - true_w) < 0.5
    ), "Gradient descent for params failed to learn correct weights"
    assert (
        np.abs(learned_b - true_b) < 0.5
    ), "Gradient descent for params failed to learn correct bias"

    print("Gradient descent for params successful\n")


def test_phase1_function1():
    print("Testing gradient descent on function: f(x) = ln(ln(exp((x - 1)^2) + 2))")

    x = Variable(value=0)
    v1 = AddConst(x, -1)
    v2 = PowConst(v1, 2)
    v3 = Exp(v2)
    v4 = AddConst(v3, 2)
    v5 = Log(v4)
    output = Log(v5)

    expected_min = 1
    min_x = gradient_descent_input(output, x, initial_x=9)
    print(f"min_x = {np.round(min_x.ravel()[0], 4)}")
    assert np.allclose(min_x, expected_min), "Gradient descent failed"

    print("Gradient descent successful\n")


def test_phase1_function2():
    print("Testing gradient descent on function: f(x) = (x^2 - 1)^3")

    x = Variable(value=0)
    v1 = PowConst(x, 2)
    v2 = AddConst(v1, -1)
    output = PowConst(v2, 3)

    expected_min = 0
    min_x = gradient_descent_input(output, x, initial_x=-0.8)
    print(f"min_x = {np.round(min_x.ravel()[0], 4)}")
    assert np.allclose(min_x, expected_min), "Gradient descent failed"

    print("Gradient descent successful\n")


def test_phase1_function3():
    print("Testing gradient descent on function: f(x) = ln(1 + exp(x^2))")

    x = Variable(value=0)
    v1 = PowConst(x, 2)
    v2 = Exp(v1)
    v3 = AddConst(v2, 1)
    output = Log(v3)

    expected_min = 0
    min_x = gradient_descent_input(output, x, initial_x=4)
    print(f"min_x = {np.round(min_x.ravel()[0], 4)}")
    assert np.allclose(min_x, expected_min), "Gradient descent failed"

    print("Gradient descent successful\n")


def test_phase1():
    test_phase1_function1()
    test_phase1_function2()
    test_phase1_function3()


def test_impact_of_iterations_on_loss():
    print("Testing gradient vs stochastic gradient")

    np.random.seed(100)

    MAX_ITERATIONS = 50000
    STEP_SIZE = 2500
    initial = 10000
    ns = []
    while initial <= MAX_ITERATIONS:
        ns.append(initial)
        initial += STEP_SIZE

    X, Y_ = data.sample_gauss_2d(3, 100)

    N, D = X.shape
    C = np.max(Y_) + 1
    Y = data.class_to_onehot(Y_)

    # Parameter initialization
    W = Parameter(shape=(C, D))
    W.initialize_with_random()

    b = Parameter(shape=(C, 1))
    b.initialize_with_zeros()

    x = Variable(X, name="x")
    y = Variable(Y, name="y")

    model = LogisticRegression(W, b)

    # Mode: affine transformation + softmax
    affine = AffineTransform(x, model.W, model.b, name="affine")
    scores = Softmax(affine, name="softmax")

    # Loss function: cross entropy
    loss = SoftmaxCrossEntropyWithLogits(scores, y, name="loss")

    losses = []

    for n in ns:
        print(f'Testing n={n}')
        model = LogisticRegression(deepcopy(W), deepcopy(b))
        # train the model
        model.train(X, Y_, loss, param_niter=n, param_delta=0.0001)

        # get the loss
        losses.append(model.loss)

    plt.plot(ns, losses)

    plt.show()





if __name__ == "__main__":
    """
    test_add()
    test_sub()
    test_mul()
    test_div()
    test_basic_arithmetic_expression()
    test_exp()
    test_log()
    test_powconst()
    test_affinetransform()
    test_matmul()
    test_sigmoid()
    test_relu()
    test_softmax()
    test_gradient_descent_input()
    test_gradient_descent_params()
    test_phase1()
    """
    test_impact_of_iterations_on_loss()