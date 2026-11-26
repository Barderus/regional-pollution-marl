"""Independent clipped PPO actor-critic for the Study 3 action space."""

import random

from study_3_pollutant_dynamics.dqn_agent import encode_dqn_observation


class PPOAgent:
    def __init__(self, state_size, action_count, seed, learning_rate=0.0003, discount_factor=0.99,
                 gae_lambda=0.95, clip_ratio=0.2, entropy_coefficient=0.01, value_coefficient=0.5,
                 update_epochs=4, minibatch_size=64, hidden_size=128):
        try:
            import torch
        except ImportError as error:
            message = "PPO requires PyTorch. Install it with: python -m pip install -e .[rl]"
            raise RuntimeError(message) from error

        self.torch = torch
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.state_size = state_size
        self.action_count = action_count
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.value_coefficient = value_coefficient
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.rng = random.Random(seed)
        self.training_states = set()
        self.training_state_count = 0
        self.rollout = []

        self.network = torch.nn.Sequential(
            torch.nn.Linear(state_size, hidden_size), torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size), torch.nn.Tanh(),
        ).to(self.device)
        self.policy_head = torch.nn.Linear(hidden_size, action_count).to(self.device)
        self.value_head = torch.nn.Linear(hidden_size, 1).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()),
            lr=learning_rate,
        )

    def _policy_and_value(self, states):
        features = self.network(states)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def act(self, observation):
        torch = self.torch
        state = encode_dqn_observation(observation)
        self.training_states.add(state)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self._policy_and_value(state_tensor)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
        return state, int(action.item()), float(distribution.log_prob(action).item()), float(value.item())

    def select_action(self, observation, deterministic=True):
        torch = self.torch
        state = encode_dqn_observation(observation)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._policy_and_value(state_tensor)
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                action = torch.distributions.Categorical(logits=logits).sample()
        return int(action.item())

    def remember(self, state, action, reward, old_log_probability, value, terminal):
        self.rollout.append((state, action, float(reward), old_log_probability, value, float(terminal)))

    def update(self):
        if not self.rollout:
            return

        torch = self.torch
        states, actions, rewards, old_log_probabilities, values, terminals = zip(*self.rollout)
        advantages = [0.0] * len(rewards)
        next_advantage = 0.0
        next_value = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            continuation = 1.0 - terminals[index]
            delta = rewards[index] + self.discount_factor * next_value * continuation - values[index]
            next_advantage = delta + self.discount_factor * self.gae_lambda * continuation * next_advantage
            advantages[index] = next_advantage
            next_value = values[index]

        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = advantages_tensor + torch.tensor(values, dtype=torch.float32, device=self.device)
        if len(advantages) > 1:
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probabilities_tensor = torch.tensor(old_log_probabilities, dtype=torch.float32, device=self.device)

        for _ in range(self.update_epochs):
            indices = list(range(len(rewards)))
            self.rng.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                batch = torch.tensor(indices[start:start + self.minibatch_size], dtype=torch.int64, device=self.device)
                logits, predicted_values = self._policy_and_value(states_tensor[batch])
                distribution = torch.distributions.Categorical(logits=logits)
                log_probabilities = distribution.log_prob(actions_tensor[batch])
                ratios = torch.exp(log_probabilities - old_log_probabilities_tensor[batch])
                unclipped = ratios * advantages_tensor[batch]
                clipped = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages_tensor[batch]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = torch.nn.functional.mse_loss(predicted_values, returns_tensor[batch])
                loss = policy_loss + self.value_coefficient * value_loss - self.entropy_coefficient * distribution.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.network.parameters()) + list(self.policy_head.parameters()) + list(self.value_head.parameters()),
                    0.5,
                )
                self.optimizer.step()

        self.training_state_count = len(self.training_states)
        self.rollout.clear()

    def save(self, filename):
        self.torch.save({"network": self.network.state_dict(), "policy_head": self.policy_head.state_dict(),
                         "value_head": self.value_head.state_dict(),
                         "training_state_count": self.training_state_count}, filename)

    def load(self, filename):
        checkpoint = self.torch.load(filename, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        self.value_head.load_state_dict(checkpoint["value_head"])
        self.training_state_count = int(checkpoint["training_state_count"])
