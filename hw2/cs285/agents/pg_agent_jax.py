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
    """Calculate discounted return for entire trajectory."""
    T = len(rewards)
    discount_factors = gamma ** np.arange(T)
    discounted_return = np.sum(rewards * discount_factors)
    return np.full_like(rewards, discounted_return)


def calculate_discounted_reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Calculate discounted reward-to-go more efficiently."""
    T = len(rewards)
    rtg = np.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(T)):
        running_sum = rewards[t] + gamma * running_sum
        rtg[t] = running_sum
    return rtg


class PGAgent:
    """Policy Gradient Agent implementation using JAX.

    This class implements a Policy Gradient agent with the following features:
    - MLP-based policy network
    - Support for both discrete and continuous action spaces
    - Optional reward-to-go formulation
    - Advantage normalization
    - Gamma discounting

    Attributes:
        actor: MLPPolicyPG network for action selection
        policy_train_state: Current training state of the policy
        gamma: Discount factor for future rewards
        use_reward_to_go: Whether to use reward-to-go formulation
        normalize_advantages: Whether to normalize advantages
        rng: JAX random number generator key
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
        # gae_lambda: float | None = None,
        normalize_advantages: bool = False,
        rng: Array = jax.random.PRNGKey(0),
    ):
        self.actor = MLPPolicyPG(
            ac_dim=ac_dim,
            ob_dim=ob_dim,
            discrete=discrete,
            n_layers=n_layers,
            layer_size=layer_size,
        )

        if use_baseline:
            self.critic = ValueCritic(ob_dim, n_layers, layer_size)
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
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory.
        The batch size is the total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """
        # Step 1: calculate Q-values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        flat_obs = np.concatenate(obs)
        flat_actions = np.concatenate(actions)
        flat_qvals = np.concatenate(q_values)
        flat_rewards = np.concatenate(rewards)
        flat_terminals = np.concatenate(terminals) if terminals is not None else None

        # Step 2: estimate advantages (possibly using the critic baseline)
        advantages, advantages_info = self._estimate_advantage(flat_obs, flat_rewards, flat_qvals, flat_terminals)
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # Step 3: update the policy
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
        """Monte Carlo estimation of Q-values."""
        q_values = []
        if not self.use_reward_to_go:
            # Trajectory-based PG: use full discounted return for all time steps in the trajectory.
            for trajectory_rewards in rewards:
                discounted_returns = calculate_discounted_return(trajectory_rewards, self.gamma)
                q_values.append(discounted_returns)
        else:
            # Reward-to-go PG: use the discounted sum of rewards from time t onward.
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
        Computes advantages by subtracting a learned baseline from the estimated Q-values.
        If a critic is used, it is updated for a number of gradient steps (baseline_gradient_steps)
        and then used to predict the state value for each observation. The advantage is then computed as:

            advantage = Q(s, a) - V(s)

        If no critic is used, the Q-values are returned directly.
        """
        # If using a critic (baseline), update it and compute advantages.
        metrics = {}
        if self.critic is not None:
            obs_jnp = jtu.from_numpy(obs)
            q_values_jnp = jtu.from_numpy(obs)

            for _ in range(self.baseline_gradient_steps):
                self.critic_train_state, loss = self.critic.update(self.critic_train_state, obs_jnp, q_values_jnp)
                metrics = {
                    "Critic Loss": loss,
                }

            baseline_predictions = self.critic.apply_fn(self.critic_train_state.params, obs_jnp)  # type: ignore
            # Compute the advantage by subtracting the baseline predictions from the Q-values.
            advantages = q_values - np.array(baseline_predictions)
            return advantages, metrics
        # No baseline used, so the advantage is simply the Q-value.
        return q_values, metrics

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample action from the policy."""
        self.rng, rng = jax.random.split(self.rng)
        return self.actor.get_action(obs, self.policy_train_state.params, rng)
