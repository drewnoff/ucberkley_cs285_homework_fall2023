from __future__ import annotations
import os
import time
from cs285.agents.pg_agent_jax import PGAgent
import gym
import numpy as np
import jax
from cs285.infrastructure import utils
from cs285.infrastructure import jax_util as ju
from cs285.infrastructure.logger import Logger
from cs285.infrastructure.action_noise_wrapper import ActionNoiseWrapper

MAX_NVIDEO = 2


def run_training_loop(args):
    logger = Logger(args.logdir)

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)

    env = gym.make(args.env_name, render_mode=None, new_step_api=True)
    discrete = isinstance(env.action_space, gym.spaces.Discrete) # type: ignore

    if args.action_noise_std > 0:
        assert not discrete, f"Cannot use --action_noise_std for discrete environment {args.env_name}"
        env = ActionNoiseWrapper(env, args.seed, args.action_noise_std)

    max_ep_len = args.ep_len or env.spec.max_episode_steps
    ob_dim = env.observation_space.shape[0] # type: ignore
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0] # type: ignore

    fps = 1 / env.model.opt.timestep if hasattr(env, "model") else env.env.metadata["render_fps"]

    agent = PGAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        learning_rate=args.learning_rate,
        use_reward_to_go=args.use_reward_to_go,
        gamma=args.discount,
        normalize_advantages=args.normalize_advantages,
        rng=rng,
    )

    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        print(f"\n********** Iteration {itr} ************")

        # collect trajectories
        obs_list = []
        action_list = []
        reward_list = []
        envsteps_this_batch = 0
        num_steps = 0

        while num_steps < args.batch_size:
            obs = env.reset()
            done = False
            trajectory_obs = []
            trajectory_actions = []
            trajectory_rewards = []
            steps = 0

            while not done and steps < max_ep_len:
                action = ju.to_numpy(agent.get_action(obs))
                try:
                    next_obs, reward, done, _, _ = env.step(action)
                except AttributeError:
                    print("Warning: issues with numpy version compatibility, skipping this step")
                    continue

                trajectory_obs.append(obs)
                trajectory_actions.append(action)
                trajectory_rewards.append(reward)

                obs = next_obs
                steps += 1
                num_steps += 1

            obs_list.append(np.array(trajectory_obs))
            action_list.append(np.array(trajectory_actions))
            reward_list.append(np.array(trajectory_rewards))
            envsteps_this_batch += steps

        total_envsteps += envsteps_this_batch

        # train agent
        train_info = agent.update(obs_list, action_list, reward_list)

        if itr % args.scalar_log_freq == 0:
            print("\nCollecting data for eval...")
            eval_obs_list = []
            eval_action_list = []
            eval_reward_list = []
            eval_steps = 0

            while eval_steps < args.eval_batch_size:
                obs = env.reset()
                done = False
                trajectory_obs = []
                trajectory_actions = []
                trajectory_rewards = []
                steps = 0

                while not done and steps < max_ep_len:
                    action = ju.to_numpy(agent.get_action(obs))
                    try:
                        next_obs, reward, done, _, _ = env.step(action)
                    except AttributeError:
                        print("Warning: issues with numpy version compatibility, skipping this step")
                        continue

                    trajectory_obs.append(obs)
                    trajectory_actions.append(action)
                    trajectory_rewards.append(reward)

                    obs = next_obs
                    steps += 1
                    eval_steps += 1

                eval_obs_list.append(np.array(trajectory_obs))
                eval_action_list.append(np.array(trajectory_actions))
                eval_reward_list.append(np.array(trajectory_rewards))

            train_returns = [np.sum(rewards) for rewards in reward_list]
            eval_returns = [np.sum(rewards) for rewards in eval_reward_list]

            logs = {
                "Train_AverageReturn": np.mean(train_returns),
                "Train_StdReturn": np.std(train_returns),
                "Eval_AverageReturn": np.mean(eval_returns),
                "Eval_StdReturn": np.std(eval_returns),
            }

            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time

            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")
            logger.flush()

        if args.video_log_freq != -1 and itr % args.video_log_freq == 0:
            print("\nCollecting video rollouts...")
            eval_video_trajs = utils.sample_n_trajectories(
                env, agent.actor, MAX_NVIDEO, max_ep_len, render=True
            )

            logger.log_trajs_as_videos(
                eval_video_trajs,
                itr,
                fps=fps,
                max_videos_to_save=MAX_NVIDEO,
                video_title="eval_rollouts",
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1000,
        help="steps collected per train iteration"
    )
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=400,
        help="steps collected per eval iteration"
    )

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    parser.add_argument("--action_noise_std", type=float, default=0)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "q2_pg_"  # keep for autograder

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = (
        logdir_prefix
        + args.exp_name
        + "_"
        + args.env_name
        + "_"
        + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    logdir = os.path.join(data_path, logdir)
    args.logdir = logdir
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    run_training_loop(args)


if __name__ == "__main__":
    main()
