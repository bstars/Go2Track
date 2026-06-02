import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from collections import namedtuple, deque
import gymnasium as gym

from helpers import Pi, V, RunningMeanStd
class PPO_Clip():
    def __init__(
            self,
            envs, 
            actor_dim,
            critic_dim,
            init_log_std,
            tanh_transformed=False,
            learning_rate=2e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None):
        
        self.num_envs = len(envs)
        self.state_dim = envs[0].observation_space.shape[0]
        self.action_dim = envs[0].action_space.shape[0]
        self.tanh_transformed = tanh_transformed
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.envs = envs
        self.states = [env.reset()[0] for i, env in enumerate(self.envs)]
        self.pi = Pi(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=actor_dim, tanh_transformed=tanh_transformed, init_log_std=init_log_std)
        self.v = V(self.state_dim, critic_dim)
        self.running_stat = RunningMeanStd(shape=[self.state_dim,])

        self.last_episode_returns = [0. for _ in range(self.num_envs)]
        self.episode_returns = [0. for _ in range(self.num_envs)]
        self.last_episode_lengths = [0. for _ in range(self.num_envs)]
        self.episode_lengths = [0. for _ in range(self.num_envs)]

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.pi.parameters(),  "lr": self.learning_rate},
                {"params": self.v.parameters(), "lr": self.learning_rate},
            ]
        )

    @torch.no_grad()
    def sample(self, i, steps):
        """
        sample transitions use the current policy, and record the estimated values using the current critic

        :param i
            type: int
            note: index of environment

        :return states
            type: torch.tensor
            shape: [num_steps, state_dim]

        :return actions
            type: torch.tensor
            shape: [num_steps, action_dim]

        :return log_probs, rewards, terminals, truncateds
            type: torch.tensor
            shape: [num_steps]

        :return values
            type: torch.tensor
            shape: [num_steps+1]

        :return truncated_next_values
            type: torch.tensor
            shape: [number of truncations]
        
        """
        states = []
        actions = []
        log_probs = []
        rewards = []
        terminals = []
        truncateds = []
        values = []
        truncated_values = []

        for _ in range(steps):
            state_th = torch.from_numpy(self.states[i]).float()
            state_th = self.running_stat.normalize(state_th)[None,:]
            value = self.v(state_th)[0].item()
            _, action, log_prob = self.pi(state_th)
            action = action[0].detach().numpy()
            log_prob = log_prob[0].item()

            state_next, reward, terminal, truncated, info = self.envs[i].step(action)

            states.append(self.states[i])
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            terminals.append(terminal)
            truncateds.append(truncated)
            values.append(value)

            self.states[i] = state_next
            self.episode_returns[i] += reward
            self.episode_lengths[i] += 1

            if truncated:
                # if truncated, bootstrap from next state
                state_th = torch.from_numpy(self.states[i]).float()
                state_th = self.running_stat.normalize(state_th)[None,:]
                value = self.v(state_th)[0].item()
                truncated_values.append(value)

            if terminal or truncated:
                self.states[i], _ = self.envs[i].reset()
                self.last_episode_returns[i] = self.episode_returns[i]
                self.last_episode_lengths[i] = self.episode_lengths[i]
                self.episode_returns[i] = 0.
                self.episode_lengths[i] = 0.
        
        state_th = torch.from_numpy(self.states[i]).float()
        state_th = self.running_stat.normalize(state_th)[None,:]
        value = self.v(state_th)[0].item()
        values.append(value)

        return torch.from_numpy(np.stack(states, axis=0)).float(), \
            torch.from_numpy(np.stack(actions, axis=0)).float(), \
            torch.from_numpy(np.array(log_probs)).float(), \
            torch.from_numpy(np.array(rewards)).float(), \
            torch.from_numpy(np.array(terminals)).bool(), \
            torch.from_numpy(np.array(truncateds)).bool(), \
            torch.from_numpy(np.array(values)).float(), \
            torch.from_numpy(np.array(truncated_values)).float()

    def lambda_returns(self, values, rewards, terminals, truncateds, truncated_values):
        advantages = torch.zeros_like(values).float()
        truncated_idx = -1

        for t in reversed(range(len(rewards))):
            terminal = terminals[t].float()

            if truncateds[t]:
                # if the episode is truncated, states[t+1] is from a new episode, bootstrap the value from truncated_values
                td_error = rewards[t] + self.gamma * (1 - terminal) * truncated_values[truncated_idx] - values[t]
                advantages[t] = td_error
                truncated_idx -= 1
            else:
                td_error = rewards[t] + self.gamma * (1 - terminal) * values[t+1] - values[t]
                advantages[t] = td_error + self.gae_lambda * self.gamma * (1 - terminal) * advantages[t+1]

        return advantages[:-1], advantages[:-1] + values[:-1]

    def train_step(self):
        # environment interaction, and compute GAEs
        states_all = []
        actions_all = []
        log_probs_all = []
        values_all = []
        advantages_all = []
        targets_all = []

        for i in range(self.num_envs):
            states, actions, log_probs, rewards, terminals, truncateds, values, truncated_values = self.sample(i, self.n_steps)
            advantages, targets = self.lambda_returns(values, rewards, terminals, truncateds, truncated_values)

            states_all.append(states)
            actions_all.append(actions)
            log_probs_all.append(log_probs)
            values_all.append(values[:-1])
            advantages_all.append(advantages)
            targets_all.append(targets)


        states_all = torch.concat(states_all)
        actions_all = torch.concat(actions_all)
        log_probs_all = torch.concat(log_probs_all)
        values_all = torch.concat(values_all)
        advantages_all = torch.concat(advantages_all)
        targets_all = torch.concat(targets_all)

        raw_states_all = states_all
        states_all = self.running_stat.normalize(raw_states_all)
        self.running_stat.update(raw_states_all)


        # actual training
        T = len(actions_all)
        kls = []
        clip_fractions = []
        early_end = False

        for e in range(self.n_epochs):
            idxs = np.random.permutation(T)
            for start in range(0, T, self.batch_size):
                mb = idxs[start:start+self.batch_size]
                states_b = states_all[mb]
                actions_b = actions_all[mb]
                log_probs_b = log_probs_all[mb]
                advantages_b = advantages_all[mb].clone()
                targets_b = targets_all[mb]
                values_b = values_all[mb]

                # minibatch advantage normalization
                advantages_b = (advantages_b - advantages_b.mean()) / (advantages_b.std() + 1e-5)

                # actor loss
                normal_new, _, _ = self.pi(states_b)
                if self.tanh_transformed:
                    log_probs_new = normal_new.log_prob(actions_b)
                    log_probs_new = log_probs_new - torch.log(1 - actions_b**2 + 1e-6)
                    log_probs_new = log_probs_new.sum(-1)
                else:
                    log_probs_new = normal_new.log_prob(actions_b).sum(-1)

                ratio = torch.exp(log_probs_new - log_probs_b)
                ratio_clipped = torch.clamp(ratio, 1-self.clip_range, 1 + self.clip_range)
                a_loss = - torch.mean(
                    torch.minimum(
                        ratio * advantages_b, ratio_clipped * advantages_b
                    )
                )

                # value loss
                v = self.v(states_b)
                v_loss = F.huber_loss(v, targets_b)

                # entropy loss
                e_loss = -normal_new.entropy().sum(-1).mean()


                # update
                loss = a_loss + self.vf_coef * v_loss + self.ent_coef * e_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), self.max_grad_norm)
                self.optimizer.step()

                # early stopping for smooth policy update
                with torch.no_grad():
                    # approx_kl = (log_probs_b - log_probs_new).mean().item() # naive approximation
                    approx_kl = (ratio - 1 - torch.log(ratio)).mean().item() # capture only the 2nd order   
                    clip_fraction = (torch.abs(1. - ratio) > self.clip_range).float().mean().item()
                    clip_fractions.append(clip_fraction)
                    kls.append(approx_kl)
                if self.target_kl is not None and approx_kl > 1. * self.target_kl:
                    early_end = True
                    break
            if early_end:
                break
        
        return a_loss.item(), v_loss.item(), e_loss.item(), e+1, np.mean(clip_fractions), np.mean(kls)


    def train(self, total_steps, print_every=10):
        

        current_steps = 0
        num_iter = 0

        while current_steps < total_steps:
            a_loss, v_loss, e_loss, e, clip_fractions, kls = self.train_step()

            current_steps += self.n_steps * self.num_envs
            num_iter += 1

            if num_iter % print_every == 0:
                print("iteration %d, %.2f%% trained, %d epoches, a_loss %.4f, v_loss %.4f, clipped %.2f,  std %.2f, epi length %.2f, epi return %.2f" % 
                    (
                        num_iter,
                        current_steps / total_steps * 100,
                        e,
                        a_loss, 
                        v_loss, 
                        np.mean(clip_fractions),
                        self.pi.log_std.exp().mean().item(),
                        np.mean(self.last_episode_lengths), 
                        np.mean(self.last_episode_returns)
                    )
                )

    def save(self, path):
        torch.save(
            {
                "actor": self.pi.state_dict(),
                "critic": self.v.state_dict(),
                "running_mean": self.running_stat.mean,
                "running_var": self.running_stat.var,
                "running_count": self.running_stat.count,
                "optim" : self.optimizer.state_dict()
            },
            path
        )
            
    def load(self, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.pi.load_state_dict(ckpt["actor"])
        self.v.load_state_dict(ckpt["critic"])
        self.optimizer.load_state_dict(ckpt["optim"])
        self.running_stat.mean.copy_(ckpt["running_mean"].to(self.running_stat.mean.device))
        self.running_stat.var.copy_(ckpt["running_var"].to(self.running_stat.var.device))
        self.running_stat.count = ckpt["running_count"]

        return self

        
        
if __name__ == '__main__':
    mode = sys.argv[1]
    actor_dim=[256, 256, 256]
    critic_dim=[256, 256, 256]

    if mode == "train":

        envs = [gym.make("BipedalWalker-v3") for _ in range(32)]
        ppo = PPO_Clip(
            envs, 
            actor_dim=actor_dim,
            critic_dim=critic_dim,
            init_log_std=-0.75,
            tanh_transformed=False,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=512*2,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.1,
            max_grad_norm=1.,
            target_kl=0.05,
        )

        # ppo.load("./bipedal.pth")

        ppo.train(2e6)
        ppo.save("./bipedal.pth")

    elif mode == "test":
        env = gym.make("BipedalWalker-v3", render_mode="human")
        ppo = PPO_Clip(
            [env], 
            actor_dim=actor_dim,
            critic_dim=critic_dim,
            init_log_std=-0.25,
            tanh_transformed=False,
            learning_rate=1e-3,
            n_steps=128,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.,
            vf_coef=1.,
            max_grad_norm=1.,
            target_kl=0.02,
        )

        ppo.load("./bipedal.pth")

        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:

            state_th = torch.from_numpy(state)
            state_th = ppo.running_stat.normalize(state_th)[None,:]
            _, action_th, _ = ppo.pi(state_th)
            action = action_th[0].detach().numpy()

            state, reward, terminal, truncated, info = env.step(action)
            episode_length += 1
            episode_reward += reward

            if terminal or truncated:
                break
        
        print(episode_length, episode_reward)