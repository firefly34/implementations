import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    # Building a feedforward neural network
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# Training Function
def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This only works with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # Create the policy network i.e. network that learns the policy
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

    # Make a function to compute the action distribution
    # For calculating action distribution, pass a state to the policy network
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # Make a function to select an action sampled from policy
    # To get the action int sample an action out of the action distribution
    def get_actions(obs):
        return get_policy(obs).sample().item()

    # Make a loss function whose gradient for the right data is the policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean

    # Make the optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # For training the policy network
    def train_one_epoch():
        batch_obs = []  # for observation
        batch_acts = []  # for actions
        batch_weights = []  # for R(tau) weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # Reset episode specific variables
        obs = env.reset()  # first observation comes from starting distribution
        done = False  # signal from environment that the episode is over
        ep_rews = []  # list of rewards acquired throughout episode

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        while True:

            # Render the environment
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # Save the observations
            batch_obs.append(obs.copy())

            # Take an action in the environment
            act = get_actions(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _ = env.step(act)

            # Save the action and the reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # If the episode is complete, save all the information about the episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # The weight for each log probability is the return
                # i.e. for each logprob(a|s) weight is R(tau)
                batch_weights += [ep_ret] * ep_len

                # Reset the episode specific variables
                obs, done, ep_rews = env.reset(), False, []

                # Stop rendering this epoch
                finished_rendering_this_epoch = True

                # End experience loop once you have enough samples
                if len(batch_obs) > batch_size:
                    break

        # Taking a single policy network update step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.float32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32)
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # Training Loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr)
