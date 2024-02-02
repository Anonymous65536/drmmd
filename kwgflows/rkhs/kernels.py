import abc

import jax.numpy as jnp
from flax import struct
from jax import vmap

from kwgflows.typing import Array


def _rescale(x: Array, scale: Array) -> Array:
    return x / scale


def _l2_norm_squared(x: Array) -> Array:
    return jnp.sum(jnp.square(x))


class base_kernel(struct.PyTreeNode, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x: Array, y: Array) -> Array:
        raise NotImplementedError

    def make_distance_matrix(self, X: Array, Y: Array) -> Array:
        return vmap(vmap(type(self).__call__, (None, None, 0)), (None, 0, None))(
            self, X, Y
        )


class gaussian_kernel(base_kernel):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-0.5 * _l2_norm_squared(_rescale(x - y, self.sigma)))


class laplace_kernel(base_kernel):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.exp(-jnp.sum(jnp.abs(_rescale(x - y, self.sigma))))


class imq_kernel(base_kernel):
    sigma: float
    c: float = 1.0
    beta: float = -0.5

    def __call__(self, x: Array, y: Array) -> Array:
        return jnp.power(
            self.c**2 + _l2_norm_squared(_rescale(x - y, self.sigma)), self.beta
        )


class negative_distance_kernel(base_kernel):
    sigma: float

    def __call__(self, x: Array, y: Array) -> Array:
        return -_l2_norm_squared(_rescale(x - y, self.sigma))


class energy_kernel(base_kernel):
    # x0: Array
    beta: float
    sigma: float
    eps: float = 1e-8

    def __call__(self, x: Array, y: Array) -> Array:
        x0 = jnp.zeros_like(x)

        pxx0 = jnp.power(_l2_norm_squared(_rescale(x - x0, self.sigma)) + self.eps, self.beta / 2)
        pyx0 = jnp.power(_l2_norm_squared(_rescale(y - x0, self.sigma)) + self.eps, self.beta / 2)
        pxy = jnp.power(_l2_norm_squared(_rescale(x - y, self.sigma)) + self.eps, self.beta / 2)

        ret = 0.5 * (pxx0 + pyx0 - pxy)
        return ret
