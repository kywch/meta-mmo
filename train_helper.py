import os
import time
import logging

import wandb
import torch
import numpy as np

import pufferlib.policy_pool as pp
from nmmo.render.replay_helper import FileReplayHelper

from reinforcement_learning import clean_pufferl

# Related to torch.use_deterministic_algorithms(True)
# See also https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def init_wandb(args, resume=True):
    if args.no_track:
        return None
    assert args.wandb.project is not None, "Please set the wandb project in config.yaml"
    assert args.wandb.entity is not None, "Please set the wandb entity in config.yaml"
    wandb_kwargs = {
        "id": args.exp_name or wandb.util.generate_id(),
        "project": args.wandb.project,
        "entity": args.wandb.entity,
        "config": {
            "cleanrl": args.train,
            "env": args.env,
            "agent_zoo": args.agent,
            "policy": args.policy,
            "recurrent": args.recurrent,
            "reward_wrapper": args.reward_wrapper,
        },
        "name": args.exp_name,
        "monitor_gym": True,
        "save_code": True,
        "resume": resume,
    }
    if args.wandb.group is not None:
        wandb_kwargs["group"] = args.wandb.group
    return wandb.init(**wandb_kwargs)


def train(args, env_creator, agent_creator):
    data = clean_pufferl.create(
        config=args.train,
        agent_creator=agent_creator,
        agent_kwargs={"args": args},
        env_creator=env_creator,
        env_creator_kwargs={"env": args.env, "reward_wrapper": args.reward_wrapper},
        vectorization=args.vectorization,
        exp_name=args.exp_name,
        track=args.track,
    )

    while not clean_pufferl.done_training(data):
        clean_pufferl.evaluate(data)
        clean_pufferl.train(data)

    print("Done training. Saving data...")
    clean_pufferl.close(data)
    print("Run complete.")


def sweep(args, env_creator, agent_creator):
    sweep_id = wandb.sweep(sweep=args.sweep, project=args.wandb.project)

    def main():
        try:
            args.exp_name = init_wandb(args).id
            if hasattr(wandb.config, "train"):
                # TODO: Add update method to namespace
                print(args.train.__dict__)
                print(wandb.config.train)
                args.train.__dict__.update(dict(wandb.config.train))
            train(args, env_creator, agent_creator)
        except Exception as e:  # noqa: F841
            import traceback

            traceback.print_exc()

    wandb.agent(sweep_id, main, count=20)


def generate_replay(args, env_creator, agent_creator, game_creator, seed=None):
    assert args.eval_model_path is not None, "eval_model_path must be set for replay generation"
    policies = pp.get_policy_names(args.eval_model_path)
    assert len(policies) > 0, "No policies found in eval_model_path"
    logging.info(f"Policies to generate replay: {policies}")

    save_dir = args.eval_model_path
    logging.info("Replays will be saved to %s", save_dir)

    if seed is not None:
        args.train.seed = seed
    logging.info("Seed: %d", args.train.seed)

    # Set the train config for replay
    args.train.num_envs = 1
    args.train.envs_per_batch = 1
    args.train.envs_per_worker = 1

    # Set the reward wrapper for replay
    args.reward_wrapper.eval_mode = True
    args.reward_wrapper.early_stop_agent_num = 0

    # Use the policy pool helper functions to create kernel (policy-agent mapping)
    # TODO: Revisit shuffle for evaluation
    args.train.pool_kernel = pp.create_kernel(
        args.env.num_agents,
        len(policies),  # , shuffle_with_seed=args.train.seed
    )

    data = clean_pufferl.create(
        config=args.train,
        agent_creator=agent_creator,
        agent_kwargs={"args": args},
        env_creator=env_creator,
        env_creator_kwargs={"env": args.env, "reward_wrapper": args.reward_wrapper},
        eval_mode=True,
        eval_model_path=args.eval_model_path,
        policy_selector=pp.AllPolicySelector(args.train.seed),
    )

    # Set up the game and replay helper
    replay_helper = FileReplayHelper()
    nmmo_env = data.pool.multi_envs[0].envs[0].env.env
    nmmo_env.realm.record_replay(replay_helper)
    nmmo_env.reset(game=game_creator(nmmo_env), seed=seed or args.train.seed)

    # Resets the env
    o, r, d, t, i, env_id, mask = data.pool.recv()  # This resets the env

    # Sanity checks for replay generation
    assert len(policies) == len(data.policy_pool.current_policies), "Policy count mismatch"
    assert len(data.policy_pool.kernel) == nmmo_env.max_num_agents, "Agent count mismatch"

    # Add the policy names to agent names
    if len(policies) > 1:
        for policy_id, samp in data.policy_pool.sample_idxs.items():
            policy_name = data.policy_pool.current_policies[policy_id]["name"].replace("_", "-")
            for idx in samp:
                agent_id = idx + 1  # agents are 0-indexed in policy_pool, but 1-indexed in nmmo
                nmmo_env.realm.players[agent_id].name += f"-({policy_name})"

    # Assign the specified task to the agents, if provided
    if args.task_to_assign is not None:
        raise NotImplementedError

        # NOTE: This is for the case where the curriculum file is provided
        # with open(args.curriculum, "rb") as f:
        #     task_with_embedding = dill.load(f)  # a list of TaskSpec
        # assert 0 <= args.task_to_assign < len(task_with_embedding), "Task index out of range"
        # select_task = task_with_embedding[args.task_to_assign]
        # tasks = make_task_from_spec(
        #     nmmo_env.possible_agents, [select_task] * len(nmmo_env.possible_agents)
        # )

        # # Reassign the task to the agents
        # nmmo_env.tasks = tasks
        # nmmo_env._map_task_to_agent()  # update agent_task_map
        # for agent_id in nmmo_env.possible_agents:
        #     # task_spec must have tasks for all agents, otherwise it will cause an error
        #     task_embedding = nmmo_env.agent_task_map[agent_id][0].embedding
        #     nmmo_env.obs[agent_id].gym_obs.reset(task_embedding)

        # print(f"All agents are assigned: {nmmo_env.tasks[0].spec_name}\n")

    # Generate the replay
    replay_helper.reset()
    while True:
        with torch.no_grad():
            o = torch.as_tensor(o)
            r = torch.as_tensor(r).float().to(data.device).view(-1)
            d = torch.as_tensor(d).float().to(data.device).view(-1)

            # env_pool must be false for the lstm to work
            next_lstm_state = data.next_lstm_state
            if next_lstm_state is not None:
                next_lstm_state = (
                    next_lstm_state[0][:, env_id],
                    next_lstm_state[1][:, env_id],
                )

            actions, logprob, value, next_lstm_state = data.policy_pool.forwards(
                o.to(data.device), next_lstm_state
            )

            if next_lstm_state is not None:
                h, c = next_lstm_state
                data.next_lstm_state[0][:, env_id] = h
                data.next_lstm_state[1][:, env_id] = c

            value = value.flatten()

            data.pool.send(actions.cpu().numpy())
            o, r, d, t, i, env_id, mask = data.pool.recv()

        num_alive = len(nmmo_env.agents)
        task_done = sum(1 for task in nmmo_env.tasks if task.completed)
        print("Tick:", nmmo_env.realm.tick, ", alive agents:", num_alive, ", task done:", task_done)
        if nmmo_env.game.is_over:
            if nmmo_env.game.winners is not None:
                print("Winners:", nmmo_env.game.winners)
            else:
                print("No winners.")
            break

    # Count how many agents completed the task
    print("--------------------------------------------------")
    print("Task:", nmmo_env.tasks[0].spec_name)
    num_completed = sum(1 for task in nmmo_env.tasks if task.completed)
    print("Number of agents completed the task:", num_completed)
    avg_progress = np.mean([task.progress_info["max_progress"] for task in nmmo_env.tasks])
    print(f"Average maximum progress (max=1): {avg_progress:.3f}")
    avg_completed_tick = 0
    if num_completed > 0:
        avg_completed_tick = np.mean(
            [task.progress_info["completed_tick"] for task in nmmo_env.tasks if task.completed]
        )
    print(f"Average completed tick: {avg_completed_tick:.1f}")

    # Save the replay file
    replay_file = f"{nmmo_env.game.name.lower()}_seed_{args.train.seed}_"
    if args.task_to_assign is not None:
        replay_file += f"task_{args.task_to_assign}_"
    replay_file = os.path.join(save_dir, replay_file + time.strftime("%Y%m%d_%H%M%S"))
    print(f"Saving replay to {replay_file}")
    replay_helper.save(replay_file, compress=True)
    clean_pufferl.close(data)

    return replay_file
