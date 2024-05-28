import os
import json
import random
import logging
import argparse
from collections import defaultdict

import numpy as np

import nmmo
import pufferlib
import pufferlib.policy_pool as pp

from reinforcement_learning import clean_pufferl, environment
from agent_zoo import baseline as default_learner
from train import get_init_args
from train_helper import make_game_creator

NUM_PVP_EVAL_EPISODE = 200
GAME_CLS = {
    "survive": environment.Survive,
    "battle": environment.TeamBattle,
    "task": environment.MultiTaskEval,
    "race": environment.RacetoCenter,
    "koh": environment.KingoftheHill,
    "sandwich": environment.Sandwich,
    "ptk": environment.ProtectTheKing,
}

ENV_CONFIG = pufferlib.namespace(
    **{
        "map_force_generation": False,
        "maps_path": "maps/eval/",
        "map_size": 128,
        "num_maps": 256,
        "max_episode_length": 1024,
        "num_agents": 128,
        "num_agents_per_team": 8,
        "resilient_population": 0,
        "spawn_immunity": 20,
        "curriculum_file_path": "curriculum/neurips_curriculum_with_embedding.pkl",
    }
)


def get_eval_config(debug=False):
    return {
        "device": "cuda",
        "num_envs": 16 if not debug else 1,
        "batch_size": 2**15 if not debug else 2**12,
    }


def make_env_creator(game, use_mini=False):
    assert game in GAME_CLS, f"Invalid game: {game}"

    def env_creator(*args, **kwargs):  # dummy args
        # args.env is provided as kwargs
        # TODO: use different map for eval by creating EvalConfig
        if use_mini is True:
            config = environment.MiniGameConfig(ENV_CONFIG)
        else:
            config = environment.FullGameConfig(ENV_CONFIG)
        config.set("TERRAIN_FLIP_SEED", True)
        config.set("GAME_PACKS", [(GAME_CLS[game], 1)])

        env = nmmo.Env(config)
        # Reward wrapper is for the learner, which is not used in evaluation
        env = default_learner.RewardWrapper(
            env,
            **{
                "eval_mode": True,
                "early_stop_agent_num": 0,
            },
        )
        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator


def make_agent_creator():
    # NOTE: Assuming all policies are recurrent, which may not be true
    policy_args = get_init_args(default_learner.Policy.__init__)
    recurrent_args = get_init_args(default_learner.Recurrent.__init__)

    def agent_creator(env, args=None):
        policy = default_learner.Policy(env, **policy_args)
        policy = default_learner.Recurrent(env, policy, **recurrent_args)
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
        return policy.to(get_eval_config()["device"])

    return agent_creator


class FixedPolicySelector(pp.PolicySelector):
    def __init__(self, seed, num_policies):
        self._rng = np.random.RandomState(seed)
        self._shuffle_idx = self._rng.permutation(num_policies)

    def __call__(self, items: list, num: int):
        assert num == len(items), "FixedPolicySelector: num must match len(items)"
        assert num == len(self._shuffle_idx), "FixedPolicySelector: num must match len(shuffle_idx)"
        # Return the items in the shuffled order, which is fixed
        return [items[i] for i in self._shuffle_idx]


class EvalRunner:
    def __init__(self, policy_store_dir, use_mini=False, debug=False):
        self.policy_store_dir = policy_store_dir
        self._use_mini = use_mini
        self._debug = debug

    def set_debug(self, debug):
        self._debug = debug

    def setup_evaluator(self, game, seed):
        policies = pp.get_policy_names(self.policy_store_dir)
        assert len(policies) > 0, "No policies found in eval_model_path"
        logging.info(f"Policies to evaluate: {policies}")

        env_creator = make_env_creator(game, use_mini=self._use_mini)
        sample_env = env_creator().env.env
        # Just to get the pool kernel
        _, pool_kernel = make_game_creator(game, len(policies), sample_env)

        config = self.get_pufferl_config(self._debug)
        config.seed = seed
        config.data_dir = self.policy_store_dir
        config.pool_kernel = pool_kernel

        vectorization = (
            pufferlib.vectorization.Serial
            if self._debug
            else pufferlib.vectorization.Multiprocessing
        )

        return clean_pufferl.create(
            config=config,
            agent_creator=make_agent_creator(),
            env_creator=env_creator,
            vectorization=vectorization,
            eval_mode=True,
            eval_model_path=self.policy_store_dir,
            policy_selector=FixedPolicySelector(seed, len(policies)),
        )

    @staticmethod
    def get_pufferl_config(debug=False):
        config = get_eval_config(debug)
        # add required configs
        config["torch_deterministic"] = True
        config["total_timesteps"] = 1_000_000_000  # arbitrarily large, but will end much earlier
        config["envs_per_batch"] = config["num_envs"]
        config["envs_per_worker"] = 1
        config["env_pool"] = False  # NOTE: critical for determinism
        config["learning_rate"] = 1e-4
        config["compile"] = False
        config["verbose"] = True  # not debug
        return pufferlib.namespace(**config)

    def perform_eval(self, game, seed, num_eval_episode, save_file_prefix):
        pufferl_data = self.setup_evaluator(game, seed)
        # this is a hack
        pufferl_data.policy_pool.mask[:] = 1  # policy_pool.mask is valid for training only

        # The policy-agent mapping is fixed with FixedPolicySelector
        policy_name = {k: v["name"] for k, v in pufferl_data.policy_pool.current_policies.items()}
        policy_agent_map = {
            policy_name[policy_id]: [
                agent_id + 1 for agent_id in agent_list
            ]  # sample_idxs is 0-indexed -> agent_id is 1-indexed
            for policy_id, agent_list in pufferl_data.policy_pool.sample_idxs.items()
        }

        game_results = []
        task_results = {}
        cnt_episode = 0
        while cnt_episode < num_eval_episode:
            _, infos = clean_pufferl.evaluate(pufferl_data)

            for pol, vals in infos.items():
                cnt_episode += sum(vals["episode_done"])
                if "game_scores" in vals:
                    for episode in vals["game_scores"]:
                        # Average the scores (i.e., max task progress) of agents with the same policy
                        # NOTE: The winning team's agents got +1 bonus, so that avg score is much higher
                        game_results.append(
                            {
                                pol_name: np.mean(
                                    [score for agent_id, score in episode if agent_id in agent_list]
                                )
                                for pol_name, agent_list in policy_agent_map.items()
                            }
                        )

                # task_results is for the task-level info, used in AgentTaskEval
                if game == "task":
                    if pol not in task_results:
                        task_results[pol] = defaultdict(list)
                    for k, v in vals.items():
                        if k == "length":
                            task_results[pol][k] += v  # length is a plain list
                        if k.startswith("curriculum"):
                            task_results[pol][k] += [vv[0] for vv in v]

            pufferl_data.sort_keys = []  # TODO: check if this solves memory leak

            print(f"\nSeed: {seed}, evaluated {cnt_episode} episodes.\n")

        file_name = f"{save_file_prefix}_{seed}.json"
        self._save_results(game_results, file_name)

        if game == "task":
            # Individual task completion info
            file_name = f"curriculum_info_{seed}.json"
            self._save_results(task_results, file_name)

        clean_pufferl.close(pufferl_data)
        return game_results, file_name

    def _save_results(self, results, file_name):
        with open(os.path.join(self.policy_store_dir, file_name), "w") as f:
            json.dump(results, f)

    def run(self, game, seed=None, num_episode=None, save_file_prefix=None):
        num_episode = num_episode or NUM_PVP_EVAL_EPISODE
        save_file_prefix = save_file_prefix or game

        if self._debug:
            num_episode = 4

        if seed is None:
            seed = random.randint(10000000, 99999999)

        logging.info(f"Evaluating {self.policy_store_dir} with {game}, seed: {seed}")

        _, file_name = self.perform_eval(game, seed, num_episode, save_file_prefix)

        print(f"Saved the result file to: {file_name}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Evaluate a policy store")
    parser.add_argument("policy_store_dir", type=str, help="Path to the policy directory")
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        default="all",
        choices="all battle survive race koh sandwich task ptk".split(),
        help="Game to evaluate/replay",
    )
    parser.add_argument("-s", "--seed", type=int, default=1, help="Random seed")
    parser.add_argument(
        "-n", "--num-episode", type=int, default=None, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "-r", "--repeat", type=int, default=1, help="Number of times to repeat the evaluation"
    )
    parser.add_argument(
        "--save-file-prefix", type=str, default=None, help="Prefix for the save file"
    )
    parser.add_argument("--use-mini", action="store_true", help="Use mini game config")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    if args.game == "all":
        game_list = list(GAME_CLS.keys())
        game_list.remove("task")  # task is only for AgentTaskEval
    elif args.game in GAME_CLS:
        game_list = [args.game]
    else:
        raise ValueError(f"Invalid game: {args.game}")

    runner = EvalRunner(args.policy_store_dir, use_mini=args.use_mini, debug=args.debug)

    for game in game_list:
        for i in range(args.repeat):
            if i > 0:
                args.seed = None  # this will sample new seed
            runner.run(game, args.seed, args.num_episode, args.save_file_prefix)
