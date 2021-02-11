import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import core

from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


class TRPOBuffer:
    """
    A buffer for storing trajectories experienced by the TRPO agent interacting
    with the environment and using GAE-Lambda for calculating the advantages
    for the state action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_index, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one time step of the agent environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # Buffer has to have room, so that you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_index, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # The next two lines implement GAE Lambda
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # The next line computes the rewards to go
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_index = self.ptr

    def get(self):
        """
            Call this at the end of an epoch to get all of the data from
            the buffer, with advantages appropriately normalized (shifted to have
            mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can use this function
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean)/adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, adv=self.adv_buf, logp=self.logp_buf)
        return {
            k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()
        }


def trpo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='trpo'):

    # Special function to avoid certain slow down form PYTORCH + MPI combo
    setup_pytorch_for_mpi()

    # Setup logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Create random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate Environment
    # Get dimensions of action and observation space
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync parameters across processes
    sync_params(ac)

    # Count variables and add the information to the logger
    var_counts = tuple(core.count_variables(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)

    # set up experiment buffer
    local_steps_per_epoch = int(steps_per_epoch /num_procs())
    buf = TRPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # TODO: set up a function for computing TRPO policy loss also get useful extra information
    # Setup function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value funtions
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set Up model saving
    logger.setup_pytorch_saver(ac)

    # TODO: Create the update function for the TRPO policy
    def update():
        return

    # Prepare for the interaction with the environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # TODO: Main Loop: Collect experience in env and update/log each epoch
