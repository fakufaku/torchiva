"""
solve

min -2 * p^H v + v^H D v - log|1 - g^H v|
"""
import torch
from torchiva.linalg import mag_sq
from torchiva.overiva_iss import overiss2_compute_background_update_vector


def cost(v, p, d, g):

    return (
        -2 * torch.sum(p * v)
        + torch.sum(v**2 * d)
        - torch.log((1 - torch.sum(g * v)) ** 2)
    )


def grad(v, p, d, g):
    lambd_a = 1 - torch.sum(g * v)
    return 2.0 * (d * v - p + g / lambd_a)


def sol(p, d, g):
    a = torch.sum(g.square() / d)
    b = 1 - torch.sum(g * p / d)
    ell = (-b + torch.sqrt(b**2 + 4 * a)) / (2.0 * a)
    sol = (p - ell * g) / d
    print("check", ell, 1.0 / (1 - torch.sum(g * sol)))
    print("check2", ell**2 * a + ell * b - 1)
    return sol


def cost_c(v, p, d, g):
    lin = -2 * torch.sum(p.conj() * v).real
    sq = torch.sum((v.real**2 + v.imag**2) * d)
    log = -torch.log(abs(1 - torch.sum(g.conj() * v)) ** 2)
    return lin + sq + log


def grad_c(v, p, d, g):
    lambd_a = (1 - torch.sum(g.conj() * v)).conj()
    return d * v - p + g / lambd_a


def sol_c(p, d, g):
    eps = 0.0

    gdg = torch.sum((g.real**2 + g.imag**2) / (d + eps))
    gdp = torch.sum(g.conj() * p / (d + eps))

    b1 = 1 - gdp
    b2 = b1.real**2 + b1.imag**2
    a = b2 * gdg

    beta = (-b2 + torch.sqrt(b2**2 + 4 * a)) / (2.0 * a)
    ell = b1 * beta

    v = (p - ell * g) / (d + eps)

    print("check", ell, 1.0 / (1 - torch.sum(g.conj() * v)).conj())
    print("check2", abs(ell) ** 2 * gdg + ell * b1.conj() - 1)
    print("check3", beta**2 * a + beta * b2 - 1)

    return v


if __name__ == "__main__":

    n = 2

    d = torch.zeros((n)).uniform_() * 0.9 + 0.1
    p = torch.zeros((n)).normal_()
    g = torch.zeros((n)).normal_()

    v = sol(p, d, g)

    print("Real")
    print("cost", cost(v, p, d, g))
    print("grad", grad(v, p, d, g))

    d = torch.zeros((n)).uniform_() * 0.9 + 0.1
    p = torch.zeros((n), dtype=torch.complex64).normal_()
    g = torch.zeros((n), dtype=torch.complex64).normal_()

    v = sol_c(p, d, g)
    print("Complex")
    print("cost", cost_c(v, p, d, g))
    print("grad", grad_c(v, p, d, g))
