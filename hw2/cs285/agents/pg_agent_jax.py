from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import jax
from cs285.infrastructure import jax_util as jtu

from cs285.networks.policies_jax import MLPPolicyPG
from cs285.networks.critics_jax import ValueCritic
from jax._src.typing import Array


def calculate_discounted_return(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the full-trajectory discounted return and repeat it for each time step.

    Args:
        rewards (np.ndarray): 1D array of rewards for one trajectory.
        gamma (float): Discount factor.

    Returns:
        np.ndarray: An array of the same shape as `rewards` where each element is the 
                    discounted return over the entire trajectory.
    """
    T = len(rewards)
    discount_factors = gamma ** np.arange(T)
    discounted_return = np.sum(rewards * discount_factors)
    return np.full_like(rewards, discounted_return)


def calculate_discounted_reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the discounted reward-to-go for each time step in the trajectory.

    Args:
        rewards (np.ndarray): 1D array of rewards for one trajectory.
        gamma (float): Discount factor.

    Returns:
        np.ndarray: Array of discounted reward-to-go values.
    """
    T = len(rewards)
    rtg = np.zeros_like(rewards)
    running_sum = 0.0
    for t in reversed(range(T)):
        running_sum = rewards[t] + gamma * running_sum
        rtg[t] = running_sum
    return rtg


class PGAgent:
    """
    Policy Gradient Agent with optional baseline (critic) implemented in JAX.

    Attributes:
        actor (MLPPolicyPG): Policy network for action selection.
        critic (Optional[ValueCritic]): Value function network for state-dependent baseline.
        policy_train_state: Current training state of the policy.
        critic_train_state: Current training state of the critic, if used.
        gamma (float): Discount factor for future rewards.
        use_reward_to_go (bool): Whether to use the reward-to-go formulation.
        normalize_advantages (bool): Whether to normalize the advantages.
        rng (Array): JAX random number generator key.
    """

    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
        use_reward_to_go: bool = False,
        gamma: float = 1.0,
        use_baseline: bool = False,
        baseline_learning_rate: float = 1e-3,
        baseline_gradient_steps: int = 1,
        normalize_advantages: bool = False,
        rng: Array = jax.random.PRNGKey(0),
    ) -> None:
        """
        Initialize the PGAgent.

        Args:
            ob_dim (int): Dimensionality of the observation space.
            ac_dim (int): Dimensionality of the action space.
            discrete (bool): Whether the action space is discrete.
            n_layers (int): Number of layers in the MLP networks.
            layer_size (int): Number of units per layer.
            learning_rate (float): Learning rate for the policy network.
            use_reward_to_go (bool): Whether to use reward-to-go formulation.
            gamma (float): Discount factor.
            use_baseline (bool): Whether to use a critic baseline.
            baseline_learning_rate (float): Learning rate for the critic.
            baseline_gradient_steps (int): Number of gradient steps to update the critic.
            normalize_advantages (bool): Whether to normalize advantages.
            rng (Array): JAX random key.
        """
        self.actor = MLPPolicyPG(
            ac_dim=ac_dim,
            ob_dim=ob_dim,
            discrete=discrete,
            n_layers=n_layers,
            layer_size=layer_size,
        )

        if use_baseline:
            self.critic: ValueCritic | None = ValueCritic(ob_dim, n_layers, layer_size)
            self.baseline_gradient_steps = baseline_gradient_steps
            self.critic_train_state = self.critic.create_train_state(rng, learning_rate=baseline_learning_rate)
        else:
            self.critic = None
            self.critic_train_state = None

        self.rng, init_rng = jax.random.split(rng)
        self.policy_train_state = self.actor.create_train_state(init_rng, learning_rate)

        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """
        Update the policy network using trajectories.

        Args:
            obs (Sequence[np.ndarray]): List of observation arrays (one per trajectory).
            actions (Sequence[np.ndarray]): List of action arrays (one per trajectory).
            rewards (Sequence[np.ndarray]): List of reward arrays (one per trajectory).
            terminals (Optional[Sequence[np.ndarray]]): List of terminal flags arrays (one per trajectory).

        Returns:
            Dict[str, Any]: Dictionary with training metrics.
        """
        # Step 1: Compute Q-values for each (s_t, a_t) in each trajectory.
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        flat_obs = np.concatenate(obs)
        flat_actions = np.concatenate(actions)
        flat_qvals = np.concatenate(q_values)
        flat_rewards = np.concatenate(rewards)
        flat_terminals = np.concatenate(terminals) if terminals is not None else None

        # Step 2: Estimate advantages (using the critic baseline if available).
        advantages, advantages_info = self._estimate_advantage(flat_obs, flat_rewards, flat_qvals, flat_terminals)
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Step 3: Update the policy.
        self.policy_train_state, info = self.actor.update(
            self.policy_train_state,
            flat_obs,
            flat_actions,
            advantages,
        )
        if advantages_info:
            info.update(advantages_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """
        Compute Monte Carlo estimates of Q-values.

        Args:
            rewards (Sequence[np.ndarray]): List of reward arrays for trajectories.

        Returns:
            Sequence[np.ndarray]: List of Q-value arrays corresponding to each trajectory.
        """
        q_values = []
        if not self.use_reward_to_go:
            # Full-trajectory return for each timestep.
            for trajectory_rewards in rewards:
                discounted_returns = calculate_discounted_return(trajectory_rewards, self.gamma)
                q_values.append(discounted_returns)
        else:
            # Reward-to-go for each timestep.
            for trajectory_rewards in rewards:
                discounted_rtg = calculate_discounted_reward_to_go(trajectory_rewards, self.gamma)
                q_values.append(discounted_rtg)
        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Estimate advantages by subtracting the critic's value estimates from the Q-values.

        If a critic is used, update it over a number of gradient steps and use its prediction as the baseline:
            advantage = Q(s, a) - V(s)

        Args:
            obs (np.ndarray): Flattened array of observations.
            rewards (np.ndarray): Flattened array of rewards.
            q_values (np.ndarray): Flattened array of Q-value estimates.
            terminals (Optional[np.ndarray]): Flattened array of terminal flags (unused in current implementation).

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]:
                - The computed advantages.
                - A dictionary of metrics (e.g., critic loss).
        """
        metrics: dict[str, Any] = {}
        if self.critic is not None:
            obs_jnp = jtu.from_numpy(obs)
            q_values_jnp = jtu.from_numpy(q_values)

            # Update the critic for a fixed number of gradient steps.
            total_loss = 0.0
            for _ in range(self.baseline_gradient_steps):
                self.critic_train_state, loss = self.critic.update(self.critic_train_state, obs_jnp, q_values_jnp)
                total_loss += loss
            average_loss = total_loss / self.baseline_gradient_steps
            metrics = {"Critic Loss": average_loss}

            # Get baseline predictions and compute advantages.
            baseline_predictions = self.critic.apply_fn(self.critic_train_state.params, obs_jnp)
            advantages_jnp = q_values_jnp - baseline_predictions
            advantages = jtu.to_numpy(advantages_jnp)
            return advantages, metrics

        # If no critic is used, advantages equal the Q-values.
        return q_values, {}

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Sample an action from the policy given an observation.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            np.ndarray: The sampled action.
        """
        self.rng, rng = jax.random.split(self.rng)
        return self.actor.get_action(obs, self.policy_train_state.params, rng)
