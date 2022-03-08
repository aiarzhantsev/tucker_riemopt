from unittest import TestCase

import numpy as np
from src import backend as back
from src import set_backend

from src.tucker import Tucker
from src.riemopt import compute_gradient_projection


class Test(TestCase):

    def testGradProjection(self):
        np.random.seed(229)

        def f_full(A):
            return (A ** 2 - A).sum()

        def f(T: Tucker):
            A = T.full()
            return (A ** 2 - A).sum()

        full_grad = back.grad(f_full, argnums=0)

        A = back.randn((4, 4, 4))
        T = Tucker.full2tuck(A)

        eucl_grad = full_grad(T.full())
        riem_grad = compute_gradient_projection(f, T)

        assert(np.allclose(back.to_numpy(eucl_grad), back.to_numpy(riem_grad.full()), atol=1e-5))


