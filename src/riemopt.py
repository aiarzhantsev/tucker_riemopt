from typing import Callable

import jax
import jax.config
jax.config.update("jax_enable_x64", True)

from tucker_riemopt.src.matrix import TuckerMatrix
from tucker_riemopt.src.tucker import Tucker
import numpy as np
import jax.numpy as jnp

@jax.jit
def group_cores(core1, core2):
    d = len(core1.shape)
    r = core1.shape

    new_core = core1
    to_concat = core2

    for i in range(d):
        to_concat = jnp.pad(to_concat, [(0, r[j]) if j == i - 1 else (0, 0) for j in range(d)], mode='constant', constant_values=0)
        new_core = jnp.concatenate([new_core, to_concat], axis=i)

    return new_core


def compute_gradient_projection(T, f, g=None, dg_dS=None, dg_dU=None):
    """
    Input
        X: tensor from manifold
        
     Output
        proj: projections of gradient onto the tangent space
    """

    if g is None:
        @jax.jit
        def g(T1, core, factors):
            new_factors = [jnp.concatenate([T1.factors[i], factors[i]], axis=1) for i in range(T1.ndim)]
            new_core = group_cores(core, T1.core)

            T = Tucker(new_core, new_factors)
            return f(T)

    if dg_dS is None:
        dg_dS = jax.grad(g, argnums=1)

    if dg_dU is None:
        dg_dU = jax.grad(g, argnums=2)

    dS = dg_dS(T, T.core, [jnp.zeros_like(T.factors[i]) for i in range(T.ndim)])
    dU = dg_dU(T, T.core, [jnp.zeros_like(T.factors[i]) for i in range(T.ndim)])
    dU = [dU[i] - T.factors[i] @ (T.factors[i].T @ dU[i]) for i in range(len(dU))]
    return Tucker(group_cores(dS, T.core), [jnp.concatenate([T.factors[i], dU[i]], axis=1) for i in range(T.ndim)])


def optimize(f, X0, maxiter=10, verbose=False):
    """
    Input
        f: function to maximize
        X0: first approximation
        maxiter: number of iterations to perform

    Output
        Xk: approximation after maxiter iterations
        errs: values of functional on each step
    """
    X = X0
    max_rank = np.max(X.rank)

    errs = []
    errs.append(f(X))

    @jax.jit
    def g(T1, core, factors):
        new_factors = [jnp.concatenate([T1.factors[i], factors[i]], axis=1) for i in range(T1.ndim)]
        new_core = group_cores(core, T1.core)

        T = Tucker(new_core, new_factors)
        return f(T)

    dg_dS = jax.grad(g, argnums=1)
    dg_dU = jax.grad(g, argnums=2)
    
    for i in range(maxiter):
        if verbose:
            print(f'Doing iteration {i+1}/{maxiter}\t Calculating gradient...\t', end='\r')
        G = compute_gradient_projection(X, f, g, dg_dS, dg_dU)

        if verbose:
            print(f'Doing itaration {i+1}/{maxiter}\t Calculating tau...\t\t', end='\r')
        tau = 0.0001
        
        if verbose:
            print(f'Doing iteration {i+1}/{maxiter}\t Calculating retraction...\t', end='\r')
        X = X + tau * G
        X = X.round(max_rank=max_rank) # retraction
        
        errs.append(f(X))
        if verbose:
            print(f'Done iteration {i+1}/{maxiter}!\t Error: {errs[-1]}' + ' ' * 50, end='\n')

    return X, errs
