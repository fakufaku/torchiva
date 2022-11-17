import torch


def zero_gradients_(*args):
    for arg in args:
        if arg.grad is not None:
            arg.grad.zero_()


def detach_(*args):
    for arg in args:
        arg.detach_()


def f(W, x):
    return torch.abs(W @ x) + torch.linalg.solve(W, x)


def f_rep(W, x, n_rep):
    for i in range(n_rep):
        x = f(W, x)
    return x


def f_rep_backward(grad_output, W, x, x0, n_rep):

    W_grad = W.new_zeros(W.shape)

    grad_output = grad_output[0]
    grad_x = grad_output

    for i in range(n_rep):

        # go back one step
        with torch.no_grad():
            x = f_rep(W, x0, n_rep - i - 1)

        zero_gradients_(W, x)
        W.requires_grad_()
        x.requires_grad_()

        with torch.enable_grad():
            x2 = f(W, x)

        gradients = torch.autograd.grad(
            outputs=[x2], inputs=[W, x], grad_outputs=[grad_x]
        )

        W_grad += gradients[0]
        grad_x = gradients[1]

        detach_(W, x)

    return W_grad, grad_x


def test_autograd():

    torch.manual_seed(0)

    n_rep = 5
    d = 4
    x = torch.arange(1, d + 1, dtype=torch.float32)

    W = torch.zeros((d, d)).normal_()
    U, s, V = torch.linalg.svd(W)
    W = U

    x.requires_grad_()
    W.requires_grad_()

    with torch.enable_grad():
        y = f_rep(W, x, n_rep)

    d = torch.sum(y)

    grad_output = torch.autograd.grad(outputs=[d], inputs=[y])

    gradient_torch = torch.autograd.grad(
        outputs=[y], inputs=[W, x], grad_outputs=grad_output
    )

    gradient_rev = f_rep_backward(grad_output, W, y, x, n_rep)

    err_0 = abs(gradient_torch[0] - gradient_rev[0]).max()
    err_1 = abs(gradient_torch[1] - gradient_rev[1]).max()
    assert err_0 < 1e-4
    assert err_1 < 1e-4
