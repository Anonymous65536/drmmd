from typing import Optional, Callable

import optax
import jax
import jax.numpy as jnp
from flax import struct
from jax import grad, random
from jax.tree_util import tree_map
from jax_tqdm import scan_tqdm

from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import Divergence, KernelizedDivergence
from kwgflows.typing import Array


class GradientFlowResult(struct.PyTreeNode):
    divergence: KernelizedDivergence
    Ys: DiscreteProbability

    def get_Yt(self, t):
        return tree_map(lambda x: x[t, :], self.Ys)

    def get_Y_all(self):
        return tree_map(lambda x: x, self.Ys)
    
    def get_velocity_field(self, t, reverse=False):
        Yt = self.get_Yt(t)
        first_variation = self.divergence.get_first_variation(Yt)
        first_variation = jax.lax.cond(
            reverse, lambda _: -first_variation, lambda _: first_variation, None
        )
        return grad(first_variation)


def gradient_flow(
    divergence: KernelizedDivergence,
    rng_key: Array,
    Y: Array,
    args
) -> GradientFlowResult:
    if args.flow != 'ula':
        inject_noise_scale = args.inject_noise_scale
    else:
        inject_noise_scale = 0.
    if args.flow == 'ula':
        add_noise_scale = args.diffusion_noise_scale # Entropy regularization
    else:
        add_noise_scale = 0.

    if args.opt == 'adam' or args.opt == 'sgd':
        if args.opt == 'adam':
            optimizer = optax.adam(learning_rate=args.step_size)
        elif args.opt == 'sgd':
            optimizer = optax.sgd(learning_rate=args.step_size)
        else:
            pass
        opt_state = optimizer.init(Y)

        @scan_tqdm(args.step_num)
        def one_step(dummy, i: Array):
            opt_state, rng_key, Y = dummy
            first_variation = divergence.get_first_variation(Y)
            velocity_field = jax.vmap(grad(first_variation))
            updates, new_opt_state = optimizer.update(
                velocity_field(Y + inject_noise_scale * random.normal(rng_key, Y.shape)), opt_state)
            Y_next = optax.apply_updates(Y, updates)
            Y_next = Y_next + random.normal(rng_key, Y_next.shape) * add_noise_scale * jnp.sqrt(args.step_size)
            rng_key, _ = random.split(rng_key)
            dummy_next = (new_opt_state, rng_key, Y_next)
            return dummy_next, Y_next

        _, trajectory = jax.lax.scan(one_step, (opt_state, rng_key, Y), jnp.arange(args.step_num))
    elif args.opt == 'lbfgs':
        from scipy.optimize import fmin_l_bfgs_b
        shape = Y.shape
        flatten_shape = [shape[0] * shape[1]]
        trajectory = []
        save_trajectory = lambda x : trajectory.append(x.reshape(shape).copy())
        objective = lambda Y : divergence(Y.reshape(shape)).mean()
        objective_grad = grad(objective)
        fmin_l_bfgs_b(objective, Y.reshape(flatten_shape), fprime=objective_grad, maxiter=args.step_num,
                      callback=save_trajectory)
        trajectory = jnp.array(trajectory)
        pause = True
    else:
        raise NotImplementedError("Only support adam, sgd and lbfgs for now.")
    return GradientFlowResult(divergence=divergence, Ys=trajectory)
