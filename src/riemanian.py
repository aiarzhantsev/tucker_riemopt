from typing import Callable

import jax
import jax.numpy as jnp
from copy import deepcopy

from tucker import Tucker

def _construct_deltas(what : Tucker, where : Tucker):
    new_factors = [where.factors[i].T @ what.factors[i] for i in range(what.ndim)]
    delta_core = Tucker(what.core, new_factors).full()
    delta_factors = []
    modes = list(jnp.arange(0, what.ndim))
    for i in range(what.ndim):
        where_core_unfolding = jnp.transpose(where.core, [modes[i], *(modes[:i] + modes[i + 1:])]) \
                                .reshape((where.core.shape[i], -1), order="F")
        pinv = jnp.linalg.pinv(where_core_unfolding)
        main_contraction = Tucker(what.core, new_factors[:i] + [what.factors[i]] + new_factors[i+1:])
        main_contraction = jnp.transpose(main_contraction.full(), [modes[i], *(modes[:i] + modes[i + 1:])]) \
            .reshape((main_contraction.shape[i], -1), order="F")
        main_contraction @= pinv
        projected_part = where.factors[i] @ (where.factors[i].T @ main_contraction)
        delta_factors.append(main_contraction - projected_part)

    return [delta_core, delta_factors]

def project(what : Tucker, where : Tucker):
    # print("what", what.shape, "where", where.shape)
    delta_core, delta_factors = _construct_deltas(what, where)
    projection = Tucker(delta_core, where.factors)
    for i in range(what.ndim):
        projection += Tucker(where.core, where.factors[:i] + [delta_factors[i]] + where.factors[i+1:])

    return projection


def grad(f: Callable):
    def _grad(x : Tucker):
        x_copy = deepcopy(x)
        _, grad_X = jax.value_and_grad(f)(x)
        return project(grad_X, x_copy)
    return _grad