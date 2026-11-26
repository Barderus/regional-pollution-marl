"""Independent Deep Q-Network agent for the Study 3 action space."""

import random
from collections import deque
from study_3_pollutant_dynamics.rl_formulation import STATE_BIN_EDGES, encode_observation


def encode_dqn_observation(observation):
    discrete_state = encode_observation(observation)
    normalized_state = []

    for feature_index, feature_name in enumerate(STATE_BIN_EDGES):
        maximum_bin = len(STATE_BIN_EDGES[feature_name])
        normalized_value = discrete_state[feature_index] / maximum_bin
        normalized_state.append(normalized_value)

    return tuple(normalized_state)


class DQNAgent:
    def __init__(self, state_size, action_count, seed, learning_rate=0.001, discount_factor=0.95,
                 epsilon=1.0, replay_buffer_size=20000, batch_size=64, learning_starts=256,
                 train_frequency=4, target_update_frequency=250, hidden_size=128):
        try:
            import torch
        except ImportError as error:
            message = "DQN requires PyTorch. Install the optional RL dependency with: python -m pip install -e \".[rl]\""
            raise RuntimeError(message) from error

        self.torch = torch
        self.torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.state_size = state_size
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_update_frequency = target_update_frequency
        self.rng = random.Random(seed)
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.training_states = set()
        self.environment_steps = 0
        self.learning_steps = 0

        self.policy_network = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_count),
        ).to(self.device)
        self.target_network = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_count),
        ).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.loss_function = torch.nn.SmoothL1Loss()

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_count)

        state_tensor = self.torch.tensor([state], dtype=self.torch.float32, device=self.device)

        with self.torch.no_grad():
            action_values = self.policy_network(state_tensor)

        return int(self.torch.argmax(action_values, dim=1).item())

    def remember(self, state, action, reward, next_state, terminal):
        experience = (tuple(state), action, float(reward), tuple(next_state), float(terminal))
        self.replay_buffer.append(experience)

    def update(self, state, action, reward, next_state, terminal):
        self.remember(state, action, reward, next_state, terminal)
        self.environment_steps += 1

        if len(self.replay_buffer) < self.learning_starts:
            return None

        if self.environment_steps % self.train_frequency != 0:
            return None

        experiences = self.rng.sample(self.replay_buffer, self.batch_size)
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []

        for experience in experiences:
            states.append(experience[0])
            actions.append(experience[1])
            rewards.append(experience[2])
            next_states.append(experience[3])
            terminals.append(experience[4])

        state_tensor = self.torch.tensor(states, dtype=self.torch.float32, device=self.device)
        action_tensor = self.torch.tensor(actions, dtype=self.torch.int64, device=self.device).unsqueeze(1)
        reward_tensor = self.torch.tensor(rewards, dtype=self.torch.float32, device=self.device)
        next_state_tensor = self.torch.tensor(next_states, dtype=self.torch.float32, device=self.device)
        terminal_tensor = self.torch.tensor(terminals, dtype=self.torch.float32, device=self.device)
        predicted_values = self.policy_network(state_tensor).gather(1, action_tensor).squeeze(1)

        with self.torch.no_grad():
            next_values = self.target_network(next_state_tensor).max(dim=1).values
            target_values = reward_tensor + self.discount_factor * next_values * (1 - terminal_tensor)

        loss = self.loss_function(predicted_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        self.optimizer.step()
        self.learning_steps += 1

        if self.learning_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

        return float(loss.item())

    def decay_epsilon(self, decay_rate, minimum):
        self.epsilon = max(minimum, self.epsilon * decay_rate)

    def save(self, filename):
        self.torch.save(self.policy_network.state_dict(), filename)

    def load(self, filename):
        state_dictionary = self.torch.load(filename, map_location=self.device, weights_only=True)
        self.policy_network.load_state_dict(state_dictionary)
        self.target_network.load_state_dict(state_dictionary)
