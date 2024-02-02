import os
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.70'
import jax
import jax.numpy as jnp
from jax import jit, random, vmap
import sys
import pwd
import argparse
import numpy as np
import time
import pickle
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_platform_name", "cpu")


from kwgflows.gradient_flow import gradient_flow
from kwgflows.divergences.mmd import *
from kwgflows.rkhs.kernels import *
from kwgflows.utils import *
from kwgflows.generate_data import *

def get_config():
    parser = argparse.ArgumentParser(description='CHARD')

    # Args settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='Gaussian')
    parser.add_argument('--flow', type=str, default=None)
    parser.add_argument('--kernel', type=str, default='Gaussian')
    parser.add_argument('--lmbda', type=float, default=1.0)
    parser.add_argument('--step_size', type=float, default=0.001) # Step size will be rescaled by lmbda, the actual step size = step size * lmbda
    parser.add_argument('--save_path', type=str, default='./results/')
    parser.add_argument('--bandwidth', type=float, default=1.0)
    parser.add_argument('--step_num', type=int, default=10000)
    parser.add_argument('--source_particle_num', type=int, default=300)
    parser.add_argument('--target_particle_num', type=int, default=300)
    parser.add_argument('--diffusion_noise_scale', type=float, default=1.0)
    parser.add_argument('--inject_noise_scale', type=float, default=0.0)
    parser.add_argument('--opt', type=str, default='sgd')
    args = parser.parse_args()  
    # args.step_size *= min(args.lmbda, 1.0) # Step size will be rescaled by lmbda
    return args

def create_dir(args):
    if args.seed is None:
        args.seed = int(time.time())
    args.save_path += f"{args.dataset}_dataset/{args.kernel}_kernel/{args.flow}_flow/"
    args.save_path += f"__lmbda_{args.lmbda}__step_size_{round(args.step_size, 8)}__bandwidth_{args.bandwidth}__step_num_{args.step_num}"
    args.save_path += f"__source_particle_num_{args.source_particle_num}__inject_noise_scale_{args.inject_noise_scale}"
    args.save_path += f"__opt_{args.opt}__seed_{args.seed}"
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/configs', 'wb') as handle:
        pickle.dump(vars(args), handle, protocol=pickle.HIGHEST_PROTOCOL)
    return args


# def create_gradient_flow_config(args, kernel):
#     config = GradientFlowConfig(flow=args.flow,step_size=args.step_size, 
#                                 num_steps=args.step_num, opt=args.opt,
#                                 kernel=kernel)
#     return config


def main(args):
    rng_key = random.PRNGKey(args.seed)
    N = args.source_particle_num
    M = args.target_particle_num
    assert N == M # Only consider Source and target particle number are the same for now

    if args.dataset == 'MixGaussian':
        mu = jnp.array([[-2., -2.], [-2., 2.], [2., -2.], [2., 2.]]).T
        std = jnp.sqrt(jnp.array([[0.3, 0.3]] * 4)).T
        X, Y, source_dist, target_dist = generate_mog_and_gaussian(args, M, N, mu, std)
    elif args.dataset == 'ThreeRing':
        X, Y = generate_three_ring_and_gaussian(args, int(M / 3), N) # Three ring, so divide by 3
    elif args.dataset == 'Student':
        X, Y = generate_student_and_gaussian(args, M, N)
    else:
        raise NotImplementedError
    
    if args.kernel == 'Gaussian':
        kernel = gaussian_kernel(args.bandwidth)
    elif args.kernel == 'Laplacian':
        kernel = laplace_kernel(args.bandwidth)
    elif args.kernel == 'Energy':
        kernel = energy_kernel(1., args.bandwidth, 1e-8)
    else:
        raise NotImplementedError
    args.kernel_fn = kernel

    if args.flow == 'mmd':
        divergence = mmd_fixed_target(kernel, X)
    elif args.flow == 'chard':
        divergence = chard_fixed_target(kernel, args.lmbda, X)
    elif args.flow == 'ula':
        if args.dataset == 'MixGaussian':
            divergence = ula(kernel, args.lmbda, X, target_dist)
        else:
            raise NotImplementedError("ULA only implemented for MixGaussian")
    else:
        raise NotImplementedError
    
    # config = create_gradient_flow_config(args, kernel)
    ret = gradient_flow(divergence, rng_key, Y, args)
    rate = 100
    evaluate(args, ret, rate)
    save_animation(args, ret, rate, save_path=args.save_path)
    pause = True


if __name__ == '__main__':
    args = get_config()
    args = create_dir(args)
    main(args)
    print('Program finished!')