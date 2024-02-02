import os
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import jax
import jax.numpy as jnp
import ot
import ott
from ott.geometry import costs, pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn

from kwgflows.divergences.mmd import *

plt.rcParams['axes.grid'] = True
# plt.rcParams['font.family'] = 'DeJavu Serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.labelsize'] = 20
# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amsmath, amsfonts}')
plt.tight_layout()

plt.rc('font', size=20)
plt.rc('lines', linewidth=2)
plt.rc('legend', fontsize=18, frameon=False)
plt.rc('xtick', labelsize=14, direction='in')
plt.rc('ytick', labelsize=14, direction='in')
plt.rc('figure', figsize=(6, 4))

FLOW_LIST = ['mmd', 'chard']

def compute_wasserstein_distance_numpy(X, Y):
    a, b = jnp.ones((X.shape[0], )) / X.shape[0], jnp.ones((Y.shape[0], )) / Y.shape[0]
    M = ot.dist(X, Y, 'euclidean')
    W = ot.emd(a, b, M)
    Wd = (W * M).sum()
    return Wd

@jax.jit
def compute_wasserstein_distance_jax(X, Y):
    """
    This is the jax implementation for computing the Wasserstein distance.
    However, it can only optimize the sinkorn divergence.
    When setting the epsilon=0.0, the optimization returns zero.
    Not sure why.
    """
    cost_fn = costs.PNormP(p=1)
    geom = pointcloud.PointCloud(X[:, None], Y[:, None], cost_fn=cost_fn, epsilon=0.001)
    ot_prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn()
    ot = solver(ot_prob)
    Wd = jnp.sum(ot.matrix * ot.geom.cost_matrix)
    return Wd


def compute_wasserstein_distance_trajectory(flow_1, flow_2, eval_freq):
    assert flow_1.shape[0] == flow_2.shape[0]
    T = flow_1.shape[0]
    wasserstein_distance = []
    for i in range(0, T, eval_freq):
        wasserstein_distance.append(compute_wasserstein_distance_numpy(flow_1[i, :], flow_2[i, :]))
    wasserstein_distance = jnp.array(wasserstein_distance)
    return wasserstein_distance

def evaluate(args, ret, rate):
    # Save the trajectory
    eval_freq = rate
    jnp.save(f'{args.save_path}/Ys.npy', ret.Ys[::eval_freq, :])

    T = ret.Ys.shape[0]
    X = ret.divergence.X

    wass_distance = compute_wasserstein_distance_trajectory(ret.Ys, jnp.repeat(X[None, :], T, axis=0), eval_freq)
    
    mmd_divergence = mmd_fixed_target(args.kernel_fn, X)
    mmd_distance = jnp.sqrt(jax.vmap(mmd_divergence)(ret.Ys[::eval_freq, :]))

    chard_divergence = chard_fixed_target(args.kernel_fn, args.lmbda, X)
    chard_distance = jax.vmap(chard_divergence)(ret.Ys[::eval_freq, :])

    fig, axs = plt.subplots(1, 4, figsize=(12, 4))
    axs[0].plot(wass_distance, label='Wass 2')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('Wasserstein 2 distance')
    axs[1].plot(mmd_distance, label='mmd')
    axs[1].set_xlabel('Iteration')
    axs[1].set_ylabel('MMD distance')
    axs[2].plot(chard_distance, label='chard')
    axs[2].set_xlabel('Iteration')
    axs[2].set_ylabel('CHARD distance')
    plt.savefig(f'{args.save_path}/distance.png')
    return 


def save_animation(args, ret, rate, save_path):
    num_timesteps = ret.Ys.shape[0]
    num_frames = max(num_timesteps // rate, 1)

    def update(frame):
        _animate_scatter.set_offsets(ret.get_Yt(frame * rate)[:, ::-1])
        return (_animate_scatter,)

    # create initial plot
    animate_fig, animate_ax = plt.subplots()
    # animate_fig.patch.set_alpha(0.)
    # plt.axis('off')
    # animate_ax.scatter(ret.Ys[:, 0], ret.Ys[:, 1], label='source')
    animate_ax.set_xlim(-2.0, 1.0)
    animate_ax.set_ylim(-1.0, 1.0)

    # awkard way to share state for now
    animate_ax.scatter(ret.divergence.X[:, 1], ret.divergence.X[:, 0], label='target')
    _animate_scatter = animate_ax.scatter(ret.get_Yt(0)[:, 1], ret.get_Yt(0)[:, 0], label='target')

    ani_kale = FuncAnimation(
        animate_fig,
        update,
        frames=num_frames,
        # init_func=init,
        blit=True,
        interval=50,
    )
    ani_kale.save(f'{save_path}/animation.mp4',
                   writer='ffmpeg', fps=20)