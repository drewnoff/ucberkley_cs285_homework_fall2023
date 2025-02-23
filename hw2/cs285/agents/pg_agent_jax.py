from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp

from cs285.networks.policies_jax import MLPPolicyPG
from cs285.networks.critics_jax import ValueCritic, DistributionalValueCritic


def calculate_discounted_return(rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Compute the full-trajectory discounted return and repeat it for each time step using JAX.
    """
    T = rewards.shape[0]
    discount_factors = gamma ** jnp.arange(T)
    discounted_return = jnp.sum(rewards * discount_factors)
    return jnp.full_like(rewards, discounted_return)


def calculate_discounted_reward_to_go(rewards: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """
    Compute the discounted reward-to-go for each time step using JAX's scan.
    """
    def scan_fn(carry, r):
        new_carry = r + gamma * carry
        return new_carry, new_carry
    # Reverse the rewards so that we scan from the last to the first timestep.
    _, rtg = jax.lax.scan(scan_fn, 0.0, rewards[::-1])
    return rtg[::-1]


def compute_gae(rewards: jnp.ndarray, v: jnp.ndarray, terminals: jnp.ndarray, gamma: float, lam: float) -> jnp.ndarray:
    """
    Compute the Generalized Advantage Estimation (GAE) using JAX's scan.
    """
    # Append a dummy V(s_{T+1}) = 0 for simpler recursive computation.
    v_extended = jnp.concatenate([v, jnp.zeros((1,), dtype=v.dtype)])

    def scan_fn(carry, inputs):
        r, vt, vt_next, term = inputs
        delta = r + gamma * vt_next * (1 - term) - vt
        new_carry = delta + gamma * lam * carry * (1 - term)
        return new_carry, new_carry

    # Reverse inputs to iterate from the end of the trajectory.
    inputs = (
        rewards[::-1],
        v_extended[:-1][::-1],
        v_extended[1:][::-1],
        terminals[::-1]
    )
    # Correctly unpack the scan outputs: (final_carry, stacked_outputs)
    _, advantages_rev = jax.lax.scan(scan_fn, 0.0, inputs)
    return advantages_rev[::-1]


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
        use_baseline: bool = False,
        baseline_learning_rate: float = 1e-3,
        baseline_gradient_steps: int = 1,
        normalize_advantages: bool = False,
        gae_lambda: float | None = None,
        rng = jax.random.PRNGKey(0),
    ) -> None:
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
            if gae_lambda is not None:
                raise ValueError("GAE (gae_lambda) requires use_baseline=True.")
            self.critic = None
            self.critic_train_state = None

        self.rng, init_rng = jax.random.split(rng)
        self.policy_train_state = self.actor.create_train_state(init_rng, learning_rate)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
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
        # Step 1: Compute Q-values for each trajectory.
        q_values = self._calculate_q_vals(rewards)

        flat_obs = jnp.concatenate([jnp.asarray(o) for o in obs])
        flat_actions = jnp.concatenate([jnp.asarray(a) for a in actions])
        flat_qvals = jnp.concatenate(q_values)
        flat_rewards = jnp.concatenate([jnp.asarray(r) for r in rewards])

        if terminals is None:
            # assuming complete episodes
            terminals = []
            for r in rewards:
                t = np.zeros_like(r, dtype=np.float32)
                t[-1] = 1.0  # mark last timestep as terminal
                terminals.append(t)
        flat_terminals = jnp.concatenate([jnp.asarray(t, dtype=jnp.float32) for t in terminals])

        # Step 2: Estimate advantages (using the state-value baseline if available).
        advantages, advantages_info = self._estimate_advantage(
            flat_obs,
            flat_rewards,
            flat_qvals,
            flat_terminals,
        )
        if self.normalize_advantages:
            advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

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

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[jnp.ndarray]:
        """
        Compute Monte Carlo estimates of Q-values.
        """
        q_values = []
        for trajectory_rewards in rewards:
            rewards_jax = jnp.asarray(trajectory_rewards)
            if not self.use_reward_to_go:
                q_val = calculate_discounted_return(rewards_jax, self.gamma)
            else:
                q_val = calculate_discounted_reward_to_go(rewards_jax, self.gamma)
            q_values.append(q_val)
        return q_values

    def _estimate_advantage(
        self,
        obs: jnp.ndarray,
        rewards: jnp.ndarray,
        q_values: jnp.ndarray,
        terminals: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, Any]]:
        metrics: dict[str, Any] = {}
        if self.critic is None:
            return q_values, metrics

        def update_critic(targets: jnp.ndarray) -> jnp.ndarray:
            total_loss = 0.0
            for _ in range(self.baseline_gradient_steps):
                self.critic_train_state, loss = self.critic.update( # type: ignore
                    self.critic_train_state, obs, targets
                )
                total_loss += loss
            return total_loss / self.baseline_gradient_steps # type: ignore

        if self.gae_lambda is not None:
            # --- GAE Advantage Estimation ---
            v = self.critic.apply(self.critic_train_state.params, obs) # type: ignore
            # v = self.critic.sample_value(obs, self.critic_train_state.params, rng) # type: ignore
            advantages = compute_gae(rewards, v, terminals, self.gamma, self.gae_lambda)
            targets = advantages + v
            metrics["Critic Loss"] = update_critic(targets)
        else:
            # --- Monte Carlo Advantage Estimation ---
            metrics["Critic Loss"] = update_critic(q_values)
            v_updated = self.critic.apply(self.critic_train_state.params, obs) # type: ignore
            # v_updated = self.critic.sample_value(obs, self.critic_train_state.params, rng) # type: ignore
            advantages = q_values - v_updated

        return advantages, metrics

    def get_action(self, obs: jnp.ndarray, rng) -> jnp.ndarray:
        """
        Sample an action from the policy given an observation.
        """
        self.rng, rng = jax.random.split(self.rng)
        return self.actor.get_action(obs, self.policy_train_state.params, rng)
