from argparse import Namespace

import dill
import numpy as np

import pufferlib
import pufferlib.emulation
from pettingzoo.utils.wrappers.base_parallel import BaseParallelWrapper

import nmmo
import nmmo.core.config as nc
import nmmo.core.game_api as ng
from nmmo import minigames
from nmmo.task.task_spec import TaskSpec, make_task_from_spec


# Goal function for AmmoTraining
def UseAnyAmmo(gs, subject, quantity: int):
    return min(subject.event.FIRE_AMMO.type.shape[0] / quantity, 1)


def get_team_dict(num_agents, num_agents_per_team):
    assert (
        num_agents % num_agents_per_team == 0
    ), "Number of agents must be divisible by number of agents per team"
    return {
        i: [i * num_agents_per_team + j + 1 for j in range(num_agents_per_team)]
        for i in range(num_agents // num_agents_per_team)
    }


class TeamBattle(ng.TeamBattle):
    _next_fog_onset = None
    _next_fog_speed = None
    _next_num_npc = None

    def set_fog_onset(self, fog_onset):
        self._next_fog_onset = fog_onset

    def set_fog_speed(self, fog_speed):
        self._next_fog_speed = fog_speed

    def set_num_npc(self, num_npc):
        self._next_num_npc = num_npc

    def _set_config(self):
        self.config.reset()
        self.config.set_for_episode("MAP_RESET_FROM_FRACTAL", True)
        self.config.set_for_episode("TERRAIN_WATER", 0.1)
        self.config.set_for_episode("TERRAIN_FOILAGE", 0.9)  # prop of stone tiles: 0.1
        self.config.set_for_episode("TERRAIN_SCATTER_EXTRA_RESOURCES", True)
        self.config.set_for_episode("DEATH_FOG_FINAL_SIZE", 4)

        fog_onset = self._next_fog_onset or self._np_random.integers(32, 256)
        fog_speed = self._next_fog_speed or 1 / self._np_random.integers(7, 12)
        self.config.set_for_episode("DEATH_FOG_ONSET", fog_onset)
        self.config.set_for_episode("DEATH_FOG_SPEED", fog_speed)

        npc_num = self._next_num_npc or self._np_random.integers(64, 256)
        self.config.set_for_episode("NPC_N", npc_num)


class AgentTraining(ng.AgentTraining):
    def _get_candidate_tasks(self, eval_mode=False):
        with open(self.config.CURRICULUM_FILE_PATH, "rb") as f:
            # curriculum file may have been changed, so read the file when sampling
            curriculum = dill.load(f)  # a list of TaskSpec

        cand_specs = [spec for spec in curriculum if spec.reward_to == "agent"]
        if eval_mode:
            cand_specs = [spec for spec in cand_specs if "eval" in spec.tags]
        else:
            cand_specs = [spec for spec in cand_specs if "eval" not in spec.tags]

        assert len(cand_specs) > 0, "There is no agent task to be sampled"
        return cand_specs

    def _make_agent_tasks(self, cand_specs):
        sampling_weights = [spec.sampling_weight for spec in cand_specs]
        sampled_spec = self._np_random.choice(
            cand_specs, size=self.config.PLAYER_N, p=sampling_weights / np.sum(sampling_weights)
        )
        return make_task_from_spec(self.config.POSSIBLE_AGENTS, sampled_spec)

    def _define_tasks(self):
        # NOTE: Some tasks may not be achievable at all when the necessary game system is off
        cand_specs = self._get_candidate_tasks(eval_mode=False)
        return self._make_agent_tasks(cand_specs)


class AgentTaskEval(AgentTraining):
    def _define_tasks(self):
        # NOTE: Some tasks may not be achievable at all when the necessary game system is off
        cand_specs = self._get_candidate_tasks(eval_mode=True)
        return self._make_agent_tasks(cand_specs)


class AmmoTraining(ng.AgentTraining):
    def is_compatible(self):
        return self.config.are_systems_enabled(["COMBAT", "EQUIPMENT", "PROFESSION"])

    def _define_tasks(self):
        ammo_tasks = [TaskSpec(eval_fn=UseAnyAmmo, eval_fn_kwargs={"quantity": 10})]
        ammo_tasks *= self.config.PLAYER_N
        return make_task_from_spec(self.config.POSSIBLE_AGENTS, ammo_tasks)


class RacetoCenter(minigames.RacetoCenter):
    # Use the same setting as the original
    pass


class EasyKingoftheHill(minigames.KingoftheHill):
    _next_fog_onset = None
    _next_fog_speed = None

    def set_fog_onset(self, fog_onset):
        self._next_fog_onset = fog_onset

    def set_fog_speed(self, fog_speed):
        self._next_fog_speed = fog_speed

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_size = 60

    def _set_config(self):
        super()._set_config()
        # make the game easier by decreasing the resource demands/penalty
        self.config.set_for_episode("RESOURCE_DEPLETION_RATE", 3)  # from 5
        self.config.set_for_episode("RESOURCE_RESILIENT_POPULATION", 1)

        fog_onset = self._next_fog_onset or self._np_random.integers(1, 128)
        fog_speed = self._next_fog_speed or 1 / self._np_random.integers(8, 20)
        self.config.set_for_episode("DEATH_FOG_ONSET", fog_onset)
        self.config.set_for_episode("DEATH_FOG_SPEED", fog_speed)


class Sandwich(minigames.Sandwich):
    _next_grass_map = None
    _next_fog_speed = None

    def set_fog_speed(self, fog_speed):
        self._next_fog_speed = fog_speed

    def set_grass_map(self, grass_map):
        self._next_grass_map = grass_map

    def _set_config(self):
        # randomly select whether to use the terrain map or grass map
        self._grass_map = self._next_grass_map
        if self._grass_map is None:
            self._grass_map = self._np_random.choice([True, False], p=[0.2, 0.8])
        super()._set_config()

        fog_speed = self._next_fog_speed or 1 / self._np_random.integers(8, 17)
        self.config.set_for_episode("DEATH_FOG_SPEED", fog_speed)


class RadioRaid(minigames.RadioRaid):
    _next_grass_map = None
    _next_goal_num_npc = None

    def set_grass_map(self, grass_map):
        self._next_grass_map = grass_map

    def set_goal_num_npc(self, num_npc):
        self._next_goal_num_npc = num_npc

    def _set_config(self):
        # randomly select whether to use the terrain map or grass map
        self._grass_map = self._next_grass_map
        if self._grass_map is None:
            self._grass_map = self._np_random.choice([True, False])
        self._goal_num_npc = self._next_goal_num_npc or self._goal_num_npc
        super()._set_config()


class MiniGameConfig(
    nc.Medium,
    nc.Terrain,
    nc.Resource,
    nc.Combat,
    nc.NPC,
    nc.Communication,
):
    """Configuration for Neural MMO."""

    def __init__(self, env_args: Namespace):
        super().__init__()

        self.set("PROVIDE_ACTION_TARGETS", True)
        self.set("PROVIDE_NOOP_ACTION_TARGET", True)
        self.set("PROVIDE_DEATH_FOG_OBS", True)
        self.set("TASK_EMBED_DIM", 16)  # Use the default hash embedding provided by env
        self.set("MAP_FORCE_GENERATION", env_args.map_force_generation)
        self.set("PLAYER_N", env_args.num_agents)
        self.set("HORIZON", env_args.max_episode_length)
        self.set("MAP_N", env_args.num_maps)
        self.set("TEAMS", get_team_dict(env_args.num_agents, env_args.num_agents_per_team))
        # self.set(
        #     "DEATH_FOG_ONSET",
        #     env_args.death_fog_tick if isinstance(env_args.death_fog_tick, int) else None,
        # )
        self.set("PATH_MAPS", f"{env_args.maps_path}/{env_args.map_size}/")
        self.set("MAP_CENTER", env_args.map_size)
        # self.set("NPC_N", env_args.num_npcs)
        self.set("NPC_LEVEL_MULTIPLIER", 0.5)  # make the high-level npcs weaker
        self.set("RESOURCE_RESILIENT_POPULATION", env_args.resilient_population)
        self.set("COMBAT_SPAWN_IMMUNITY", env_args.spawn_immunity)

        self.set("CURRICULUM_FILE_PATH", env_args.curriculum_file_path)


class FullGameConfig(
    MiniGameConfig,
    nc.Progression,
    nc.Item,
    nc.Equipment,
    nc.Profession,
    nc.Exchange,
):
    pass


def make_env_creator(
    reward_wrapper_cls: BaseParallelWrapper,
    train_flag: str = None,
    use_mini: bool = False,
):
    if train_flag is None:
        game_packs = [
            (TeamBattle, 1),
            (RacetoCenter, 1),
            (EasyKingoftheHill, 1),
            (Sandwich, 1),
        ]
    elif train_flag == "tb_only":
        game_packs = [(TeamBattle, 1)]
    elif train_flag == "rc_only":
        game_packs = [(RacetoCenter, 1)]
    elif train_flag == "kh_only":
        game_packs = [(EasyKingoftheHill, 1)]
    elif train_flag == "sw_only":
        game_packs = [(Sandwich, 1)]
    elif train_flag == "no_race":
        game_packs = [(TeamBattle, 1), (EasyKingoftheHill, 1), (Sandwich, 1)]
    elif train_flag == "tb_ammo":
        game_packs = [(TeamBattle, 5), (AmmoTraining, 1)]
    elif train_flag == "tb_curr":
        game_packs = [(TeamBattle, 1), (AgentTraining, 1)]
    else:
        raise ValueError(f"Invalid train_flag: {train_flag}")

    def env_creator(*args, **kwargs):
        """Create an environment."""
        # args.env is provided as kwargs
        if use_mini is True:
            config = MiniGameConfig(kwargs["env"])
        else:
            config = FullGameConfig(kwargs["env"])
        config.set("GAME_PACKS", game_packs)

        env = nmmo.Env(config)
        env = reward_wrapper_cls(env, **kwargs["reward_wrapper"])
        env = pufferlib.emulation.PettingZooPufferEnv(env)
        return env

    return env_creator
