import os
import jax.numpy as jnp
from numpyro import distributions as np_distributions
import jax


def generate_mog_and_gaussian(args, Nx, Ny, mu, std):
    mixture_probs = jnp.array([0.25, 0.25, 0.25, 0.25])
    target_dist = np_distributions.MixtureSameFamily(
        mixing_distribution=np_distributions.Categorical(probs=mixture_probs),
        component_distribution=np_distributions.Normal(mu, std)
    )

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, _ = jax.random.split(rng_key)
    X = target_dist.sample(rng_key, (Nx,))

    rng_key, _ = jax.random.split(rng_key)
    source_dist = np_distributions.Independent(np_distributions.Normal(jnp.zeros((2,)), 0.1 * jnp.ones((2,))), 1)
    Y = source_dist.sample(rng_key, (Ny,))
    return X, Y, source_dist, target_dist


def generate_three_ring_and_gaussian(args, Nx, Ny):
    rng_key = jax.random.PRNGKey(args.seed)
    r, _delta = 0.3, 0.5
    
    X = jnp.c_[r * jnp.cos(jnp.linspace(0, 2 * jnp.pi, Nx + 1)), r * jnp.sin(jnp.linspace(0, 2 * jnp.pi, Nx + 1))][:-1]  # noqa
    for i in [1, 2]:
        X = jnp.r_[X, X[:Nx, :]-i*jnp.array([0, (2 + _delta) * r])]

    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.normal(rng_key, (Ny, 2)) / 100 - jnp.array([0, r])
    return X, Y


def generate_student_and_gaussian(args, Nx, Ny):
    freedom = 2
    rng_key = jax.random.PRNGKey(args.seed)
    
    from scipy.stats import t

    # Sample from two independent t-distributions
    X1 = jnp.array(t.rvs(freedom, size=Nx))
    X2 = jnp.array(t.rvs(freedom, size=Nx))

    # Stack the samples into a 2D array
    samples = jnp.vstack((X1, X2)).T

    # Define a linear transformation matrix A and a translation vector b
    theta = jnp.pi / 4  # 45 degree rotation
    A = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)]])

    X = samples.dot(A)
    # Filter out the samples outside [-threshold, threshold]
    threshold = 10
    mask = (X[:, 0] >= -threshold) & (X[:, 0] <= threshold) & (X[:, 1] >= -threshold) & (X[:, 1] <= threshold)
    X = X[mask]

    rng_key, _ = jax.random.split(rng_key)
    Y = jax.random.normal(rng_key, (Ny, 2)) + jnp.array([2.5, 2.5])
    return X, Y