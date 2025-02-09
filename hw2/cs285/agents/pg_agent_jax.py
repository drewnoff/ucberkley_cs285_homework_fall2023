from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import jax

from cs285.networks.policies_jax import MLPPolicyPG

# from cs285.networks.critics_jax import ValueCritic
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
        # use_baseline: bool = False,
        # baseline_learning_rate: float | None = None,
        # baseline_gradient_steps: int | None = None,
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
        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        flat_obs = np.concatenate(obs)
        flat_actions = np.concatenate(actions)
        flat_qvals = np.concatenate(q_values)
        flat_rewards = np.concatenate(rewards)
        flat_terminals = np.concatenate(terminals) if terminals is not None else None

        advantages = flat_qvals
        advantages: np.ndarray = self._estimate_advantage(
            flat_obs, flat_rewards, flat_qvals, flat_terminals
        )
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        self.policy_train_state, info = self.actor.update(
            self.policy_train_state,
            flat_obs,
            flat_actions,
            advantages,
        )

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of Q values."""
        q_values = []

        if not self.use_reward_to_go:
            # Case 1: trajectory-based PG
            for trajectory_rewards in rewards:
                discounted_returns = calculate_discounted_return(trajectory_rewards, self.gamma)
                q_values.append(discounted_returns)
        else:
            # Case 2: reward-to-go PG
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
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        # TODO actual implementation
        return q_values

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample action from policy."""
        self.rng, rng = jax.random.split(self.rng)
        return self.actor.get_action(obs, self.policy_train_state.params, rng)
