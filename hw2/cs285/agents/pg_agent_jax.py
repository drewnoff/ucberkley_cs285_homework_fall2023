from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from cs285.infrastructure import jax_util as jtu

from cs285.networks.policies_jax import MLPPolicyPG
from cs285.networks.critics_jax import ValueCritic


def calculate_discounted_return(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the full-trajectory discounted return and repeat it for each time step.
    """
    T = len(rewards)
    discount_factors = gamma ** np.arange(T)
    discounted_return = np.sum(rewards * discount_factors)
    return np.full_like(rewards, discounted_return)


def calculate_discounted_reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """
    Compute the discounted reward-to-go for each time step in the trajectory.
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
    Policy Gradient Agent with state-value baseline implemented in JAX.
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
        use_baseline: bool = True,  # baseline now always used
        baseline_learning_rate: float = 1e-3,
        baseline_gradient_steps: int = 1,
        normalize_advantages: bool = False,
        rng = jax.random.PRNGKey(0),
    ) -> None:
        """
        Initialize the PGAgent.
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
        """
        # Step 1: Compute Q-values for each (s_t, a_t) in each trajectory.
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        flat_obs = jtu.from_numpy(np.concatenate(obs))
        flat_actions = jtu.from_numpy(np.concatenate(actions))
        flat_qvals = jtu.from_numpy(np.concatenate(q_values))
        flat_rewards = jtu.from_numpy(np.concatenate(rewards))

        # # If terminals are not provided, derive them for each trajectory.
        # if terminals is None:
        #     terminals = []
        #     for r in rewards:
        #         t = np.zeros_like(r, dtype=np.float32)
        #         t[-1] = 1.0
        #         terminals.append(t)
        # flat_terminals = np.concatenate(terminals)

        # Step 2: Estimate advantages (using the state-value baseline).
        advantages, advantages_info = self._estimate_advantage(
            flat_obs,
            flat_rewards,
            flat_qvals,
            # flat_terminals
        )
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
        obs: jnp.ndarray,
        rewards: jnp.ndarray,
        q_values: jnp.ndarray,
#        terminals: np.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        metrics: dict[str, Any] = {}
        if not self.critic:
            return q_values, metrics

        def update_critic(targets: jnp.ndarray) -> float:
            total_loss = 0.0
            for _ in range(self.baseline_gradient_steps):
                self.critic_train_state, loss = self.critic.update( # type: ignore
                    self.critic_train_state, obs, targets
                )
                total_loss += loss
            return total_loss / self.baseline_gradient_steps

        # 1) Train the critic to predict the same returns used in advantage
        metrics["Critic Loss"] = update_critic(q_values)

        # 2) Recompute the baseline after updating the critic
        v_s_updated = self.critic.apply(self.critic_train_state.params, obs) # type: ignore

        # 3) Advantage = Q(s) - V(s)
        advantages_jnp = q_values - v_s_updated

        return jtu.to_numpy(advantages_jnp), metrics

    def get_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Sample an action from the policy given an observation.

        Args:
            obs (np.ndarray): The current observation.

        Returns:
            np.ndarray: The sampled action.
        """
        self.rng, rng = jax.random.split(self.rng)
        return self.actor.get_action(obs, self.policy_train_state.params, rng)


    # def _estimate_advantage(
    #     self,
    #     obs: np.ndarray,
    #     rewards: np.ndarray,
    #     q_values: np.ndarray,
    #     terminals: np.ndarray,
    # ) -> tuple[np.ndarray, dict[str, Any]]:
    #     """
    #     Estimate advantages using the state-value function baseline.
    #     """
    #     metrics: dict[str, Any] = {}
    #     obs_jnp, q_values_jnp = jtu.from_numpy(obs), jtu.from_numpy(q_values)
    #     if not self.critic:
    #         return q_values, metrics

    #     def update_critic(targets: jnp.ndarray) -> float:
    #         total_loss = 0.0
    #         for _ in range(self.baseline_gradient_steps):
    #             self.critic_train_state, loss = self.critic.update(
    #                 self.critic_train_state, obs_jnp, targets
    #             )
    #             total_loss += loss
    #         return total_loss / self.baseline_gradient_steps

    #     # Compute next observations: for terminal states, reuse the current observation.
    #     next_obs_jnp = jtu.from_numpy(
    #         np.where(terminals[:-1, None] == 0, obs[1:], obs[:-1])
    #     )
    #     next_obs_jnp = jnp.vstack([next_obs_jnp, obs[-1]])  # Ensure the last obs is copied

    #     rewards_jnp = jtu.from_numpy(rewards)
    #     terminals_jnp = jtu.from_numpy(terminals).astype(jnp.float32)

    #     # Step 1: Compute TD targets using current critic estimates for next states.
    #     v_next = self.critic.apply(self.critic_train_state.params, next_obs_jnp)
    #     td_target = rewards_jnp + self.gamma * v_next * (1.0 - terminals_jnp)

    #     # Step 2: Update critic parameters using the computed TD targets.
    #     metrics["Critic Loss"] = update_critic(td_target)

    #     # Step 3: Recompute the value estimates with the updated critic parameters.
    #     v_s_updated = self.critic.apply(self.critic_train_state.params, obs_jnp)
    #     advantages_jnp = q_values_jnp - v_s_updated

    #     return jtu.to_numpy(advantages_jnp), metrics

