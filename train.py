# from pdb import set_trace as T
import importlib
import argparse
import inspect
import logging
import yaml
import time
import sys

import pufferlib
import pufferlib.utils

from reinforcement_learning import environment
from train_helper import init_wandb, train, sweep, generate_replay

BASELINE_CURRICULUM = "curriculum/neurips_curriculum_with_embedding.pkl"


def load_from_config(agent, debug=False):
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    default_keys = (
        "env train policy recurrent sweep_metadata sweep_metric sweep wandb reward_wrapper".split()
    )
    defaults = {key: config.get(key, {}) for key in default_keys}

    debug_config = config.get("debug", {}) if debug else {}
    agent_config = config[agent]

    combined_config = {}
    for key in default_keys:
        agent_subconfig = agent_config.get(key, {})
        debug_subconfig = debug_config.get(key, {})
        combined_config[key] = {**defaults[key], **agent_subconfig, **debug_subconfig}

    return pufferlib.namespace(**combined_config)


def get_init_args(fn):
    if fn is None:
        return {}
    sig = inspect.signature(fn)
    args = {}
    for name, param in sig.parameters.items():
        if name in ("self", "env", "policy"):
            continue
        if name in ("agent_id", "is_multiagent"):  # Postprocessor args
            continue
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            continue
        elif param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        else:
            args[name] = param.default if param.default is not inspect.Parameter.empty else None
    return args


# Return env_creator, agent_creator
def setup_agent(module_name, train_flag=None, use_mini=None):
    try:
        agent_module = importlib.import_module(f"agent_zoo.{module_name}")
    except ModuleNotFoundError:
        raise ValueError(f"Agent module {module_name} not found under the agent_zoo directory.")

    env_creator = environment.make_env_creator(
        reward_wrapper_cls=agent_module.RewardWrapper,
        train_flag=train_flag,
        use_mini=use_mini,
    )

    recurrent_policy = getattr(agent_module, "Recurrent", None)

    def agent_creator(env, args):
        policy = agent_module.Policy(env, **args.policy)
        if not args.no_recurrence and recurrent_policy is not None:
            policy = recurrent_policy(env, policy, **args.recurrent)
            policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
        else:
            policy = pufferlib.frameworks.cleanrl.Policy(policy)
        return policy.to(args.train.device)

    init_args = {
        "policy": get_init_args(agent_module.Policy.__init__),
        "recurrent": get_init_args(agent_module.Recurrent.__init__)
        if recurrent_policy is not None
        else {},
        "reward_wrapper": get_init_args(agent_module.RewardWrapper.__init__),
    }

    return agent_module, env_creator, agent_creator, init_args


def combine_config_args(parser, args, config):
    clean_parser = argparse.ArgumentParser(parents=[parser])
    for name, sub_config in config.items():
        args[name] = {}
        for key, value in sub_config.items():
            data_key = f"{name}.{key}"
            cli_key = f"--{data_key}".replace("_", "-")
            if isinstance(value, bool) and value is False:
                parser.add_argument(cli_key, default=value, action="store_true")
                clean_parser.add_argument(cli_key, default=value, action="store_true")
            elif isinstance(value, bool) and value is True:
                data_key = f"{name}.no_{key}"
                cli_key = f"--{data_key}".replace("_", "-")
                parser.add_argument(cli_key, default=value, action="store_false")
                clean_parser.add_argument(cli_key, default=value, action="store_false")
            else:
                parser.add_argument(cli_key, default=value, type=type(value))
                clean_parser.add_argument(cli_key, default=value, metavar="", type=type(value))

            args[name][key] = getattr(parser.parse_known_args()[0], data_key)
        args[name] = pufferlib.namespace(**args[name])

    clean_parser.parse_args(sys.argv[1:])
    return args


def update_args(args, mode=None):
    args = pufferlib.namespace(**args)

    args.track = not args.no_track
    args.env.curriculum_file_path = args.curriculum

    vec = args.vectorization
    if vec == "serial" or args.debug:
        args.vectorization = pufferlib.vectorization.Serial
    elif vec == "multiprocessing":
        args.vectorization = pufferlib.vectorization.Multiprocessing
    elif vec == "ray":
        args.vectorization = pufferlib.vectorization.Ray
    else:
        raise ValueError("Invalid --vectorization (serial/multiprocessing/ray).")

    # TODO: load the trained baseline from wandb
    # elif args.baseline:
    #     args.track = True
    #     version = '.'.join(pufferlib.__version__.split('.')[:2])
    #     args.exp_name = f'puf-{version}-nmmo'
    #     args.wandb_group = f'puf-{version}-baseline'
    #     shutil.rmtree(f'experiments/{args.exp_name}', ignore_errors=True)
    #     run = init_wandb(args, resume=False)
    #     if args.mode == 'evaluate':
    #         model_name = f'puf-{version}-nmmo_model:latest'
    #         artifact = run.use_artifact(model_name)
    #         data_dir = artifact.download()
    #         model_file = max(os.listdir(data_dir))
    #         args.eval_model_path = os.path.join(data_dir, model_file)

    if mode in ["evaluate", "replay"]:
        assert args.eval_model_path is not None, "Eval mode requires a path to checkpoints"
        args.track = False
        # Disable env pool - see the comment about next_lstm_state in clean_pufferl.evaluate()
        args.train.env_pool = False
        args.env.resilient_population = 0
        args.reward_wrapper.eval_mode = True
        args.reward_wrapper.early_stop_agent_num = 0

    if mode == "replay":
        args.train.num_envs = args.train.envs_per_worker = args.train.envs_per_batch = 1
        args.vectorization = pufferlib.vectorization.Serial

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Parse environment argument", add_help=False)
    parser.add_argument(
        "-m", "--mode", type=str, default="train", choices="train sweep replay".split()
    )
    parser.add_argument("-a", "--agent", type=str, default="baseline", help="Agent module to use")
    parser.add_argument(
        "-t", "--train-flag", type=str, default=None, help="Training game pack flag"
    )
    parser.add_argument(
        "-n", "--exp-name", type=str, default=None, help="Need exp name to resume the experiment"
    )
    parser.add_argument(
        "-p", "--eval-model-path", type=str, default=None, help="Path to model to evaluate"
    )
    parser.add_argument(
        "-g",
        "--game",
        type=str,
        default=None,
        choices="survive battle task ptk race koh sandwich".split(),
        help="Game to evaluate/replay",
    )
    parser.add_argument(
        "-c", "--curriculum", type=str, default=BASELINE_CURRICULUM, help="Path to curriculum file"
    )
    # parser.add_argument(
    #     "-t",
    #     "--task-to-assign",
    #     type=int,
    #     default=None,
    #     help="The index of the task to assign in the curriculum file",
    # )
    # parser.add_argument('--baseline', action='store_true', help='Baseline run')
    parser.add_argument(
        "--vectorization",
        type=str,
        default="multiprocessing",
        choices="serial multiprocessing ray".split(),
    )
    parser.add_argument("--use-mini", action="store_true", help="Use mini game config")
    parser.add_argument("--no-recurrence", action="store_true", help="Do not use recurrence")
    parser.add_argument("--no-track", action="store_true", help="Do NOT track on WandB")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_known_args()[0].__dict__
    config = load_from_config(args["agent"], debug=args.get("debug", False))
    agent_module, env_creator, agent_creator, init_args = setup_agent(
        args["agent"], args["train_flag"], args["use_mini"]
    )

    # Update config with environment defaults
    config.policy = {**init_args["policy"], **config.policy}
    config.recurrent = {**init_args["recurrent"], **config.recurrent}
    config.reward_wrapper = {**init_args["reward_wrapper"], **config.reward_wrapper}

    # Generate argparse menu from config
    args = combine_config_args(parser, args, config)

    # Perform mode-specific updates
    args = update_args(args, mode=args["mode"])

    if args.train.env_pool is True:
        logging.warning(
            "Env_pool is enabled. This may increase training speed but break determinism."
        )

    if args.track:
        args.exp_name = init_wandb(args).id
    else:
        args.exp_name = f"nmmo_{time.strftime('%Y%m%d_%H%M%S')}"

    if args.mode == "train":
        train(args, env_creator, agent_creator)
        exit(0)
    elif args.mode == "sweep":
        sweep(args, env_creator, agent_creator)
        exit(0)
    elif args.mode == "replay":
        generate_replay(args, env_creator, agent_creator)
        exit(0)
    else:
        raise ValueError("Mode must be one of train, sweep, or evaluate")
