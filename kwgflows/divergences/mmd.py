from flax import struct
import jax.numpy as jnp
import jax
from functools import partial

from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import KernelizedDivergence
from kwgflows.rkhs.kernels import base_kernel
from kwgflows.rkhs.rkhs import rkhs_element
from kwgflows.typing import Array, Scalar, Distribution
from typing import Callable


class mmd(struct.PyTreeNode):
    kernel: base_kernel
    
    def get_witness_function(
        self, z, X, Y
    ) -> Scalar:
        z = z[None, :]
        K_zX = self.kernel.make_distance_matrix(z, X)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.get_witness_function, X=X, Y=Y)

    def __call__(self, X, Y) -> Scalar:
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

class mmd_fixed_target(struct.PyTreeNode):
    kernel: base_kernel
    X: Array # Target samples
    
    def get_witness_function(
        self, z, Y
    ) -> Scalar:
        z = z[None, :]
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, Y) -> Callable:
        return partial(self.get_witness_function, Y=Y)

    def __call__(self, Y) -> Scalar: # mmd^2
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()


class chard(struct.PyTreeNode):
    kernel: base_kernel
    lmbda: float
    
    def witness_function(
        self, z, X, Y
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, X)
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + N * self.lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ inv_K_XX @ K_XY).mean(axis=1)
        part3 = K_zX @ inv_K_XX @ K_XX.mean(axis=1)
        return (part1 + part2 + part3).squeeze() / self.lmbda * 2 * (1 + self.lmbda)
    
    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.witness_function, X=X, Y=Y)

    def __call__(self, X, Y) -> Scalar: # chard
        N, M = Y.shape[0], X.shape[0]
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + N * self.lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ inv_K_XX @ K_XY).mean()
        part3 = (K_XX.T @ inv_K_XX @ K_XY).mean() * 2
        part4 = -(K_XX.T @ inv_K_XX @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / self.lmbda * (1 + self.lmbda)


class chard_fixed_target(struct.PyTreeNode):
    kernel: base_kernel
    lmbda: float
    X: Array # Target samples

    def witness_function(
        self, z, Y
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + N * self.lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ inv_K_XX @ K_XY).mean(axis=1)
        part3 = K_zX @ inv_K_XX @ K_XX.mean(axis=1)
        return (part1 + part2 + part3).squeeze() / self.lmbda * 2 * (1 + self.lmbda)
    
    def get_first_variation(self, Y) -> Callable:
        return partial(self.witness_function, Y=Y)

    def __call__(self, Y) -> Scalar:
        N, M = Y.shape[0], self.X.shape[0]
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + M * self.lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ inv_K_XX @ K_XY).mean()
        part3 = (K_XX.T @ inv_K_XX @ K_XY).mean() * 2
        part4 = -(K_XX.T @ inv_K_XX @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / self.lmbda * (1 + self.lmbda)
    

class ula(struct.PyTreeNode):
    kernel: base_kernel
    lmbda: float
    X: Array # Target samples
    target_dist: Distribution

    def witness_function(
        self, z
        # In ULA, Y is not needed.
    ) -> Scalar:
        # log_p = self.X.shape[0] * jnp.log(self.std) + jnp.sum(-0.5 * (z - self.mu) ** 2 / self.std ** 2, axis=1)
        log_p = self.target_dist.log_prob(z).sum()
        return -log_p # Energy is negative log density
    
    def get_first_variation(self, Y) -> Callable:
        return self.witness_function
