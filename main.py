import torch
import warnings
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from game import CarRacing
from utils import clear_dir
from wrappers import *

warnings.filterwarnings("ignore", category=UserWarning)

Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):

        done_reward = None
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            with torch.set_grad_enabled(False):
                q_vals_v = net(state_v).detach()
            _, act_v = torch.max(q_vals_v, dim=1)
            action = act_v.item()

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


def make_env():
    env = CarRacing()
    env = SkipAndStep(env, skip=4)
    env = ProcessFrame(env)
    env = BufferWrapper(env, 6)
    env = ImageToPyTorch(env)
    env = ScaledFloatFrame(env)
    return env


if __name__ == '__main__':

    MEAN_REWARD_BOUND = 19.0

    gamma = 0.9
    batch_size = 32
    replay_size = 5000
    learning_rate = 3e-4
    sync_target_frames = 200
    replay_start_size = 1000

    eps_start = 0.05
    eps_decay = .99999
    eps_min = 0.05

    # Initialize environment
    env = make_env()

    record_video = True
    clear_dir("./records")
    if record_video:
        from gym.wrappers import Monitor

        env = Monitor(env, "./records/videos", force=True)

    model = DQN(env.observation_space.shape, env.action_space.n)
    # model.load_state_dict(torch.load("best_model.dat"))
    with torch.no_grad():
        target_model = DQN(env.observation_space.shape, env.action_space.n)
    for param in target_model.parameters():
        param.requires_grad = False
    buffer = ExperienceReplay(replay_size)
    agent = Agent(env, buffer)

    epsilon = eps_start

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    total_rewards = []
    losses = []
    frame_idx = 0

    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(epsilon * eps_decay, eps_min)

        reward = agent.play_step(model, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-15:])

            print("%d:  %d games, mean reward %.3f, (epsilon %.2f)" % (frame_idx, len(total_rewards), mean_reward, epsilon))

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(model.state_dict(), "best_model.dat")
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward))

            if len(buffer) < replay_start_size:
                continue

            l = 0
            for i in range(16):
                model.zero_grad()
                batch = buffer.sample(batch_size)
                states, actions, rewards, dones, next_states = batch
                states_v = torch.tensor(states)
                next_states_v = torch.tensor(next_states, requires_grad=False)
                actions_v = torch.tensor(actions, dtype=torch.int64)
                rewards_v = torch.tensor(rewards, requires_grad=False)
                done_mask = torch.ByteTensor(dones)

                state_action_values = model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
                next_state_values = target_model(next_states_v).max(1)[0].detach()
                next_state_values[done_mask] = 0.0
                expected_state_action_values = next_state_values * gamma + rewards_v

                loss_t = loss_function(state_action_values, expected_state_action_values)
                l += loss_t.detach()

                optimizer.zero_grad()
                loss_t.backward()
                optimizer.step()

            print("average loss: %.6f" % (l / 16))
            losses.append(l / 16)

        if frame_idx % sync_target_frames == 0:
            with torch.no_grad():
                target_model.load_state_dict(model.state_dict())
                np.save("rewards", np.array(total_rewards))
                np.save("losses", np.array(losses))
    env.close()
